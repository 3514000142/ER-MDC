import copy
import os
import resource
import json
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
from collections import OrderedDict # 用于排序字典

from m3ae.config import ex
from m3ae.datamodules.multitask_datamodule import MTDataModule
from m3ae.modules import M3AETransformerSS

# --- 导入位置编码函数 ---
from m3ae.modules.vision_encoders.clip_model import adapt_position_encoding
from m3ae.modules.vision_encoders.swin_helpers import swin_adapt_position_encoding
# --- 导入结束 ---


# 设置资源限制
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@torch.no_grad()
def extract_features(datamodule, model, split='train'):
    """
    遍历指定的数据划分（train/test），提取所有样本的
    multi_modal_cls_feats, qid, 和 answers (仅训练集)。
    """
    
    dataloader = None
    if split == 'train':
        dataloader = torch.utils.data.DataLoader(
            datamodule.train_dataset,
            batch_size=datamodule.batch_size,
            sampler=None,
            num_workers=datamodule.num_workers,
            collate_fn=datamodule.collate,
            shuffle=False # 必须为 False 以匹配 QID 顺序
        )
    elif split == 'test':
        dataloader = torch.utils.data.DataLoader(
            datamodule.test_dataset,
            batch_size=datamodule.batch_size,
            sampler=None,
            num_workers=datamodule.num_workers,
            collate_fn=datamodule.collate,
            shuffle=False # 必须为 False
        )
    else:
        raise ValueError(f"不支持的划分: {split}")

    all_features = []
    all_qids = []
    all_answers = [] # <-- [新] 收集答案

    print(f"为 '{split}' 划分提取特征 (shuffle=False)。")
    for batch in tqdm(dataloader, desc=f"正在提取 {split} 特征"):
        # 1. 将 Batch 移动到设备
        batch_device = {}
        for k_batch, v_batch in batch.items():
            if isinstance(v_batch, torch.Tensor):
                batch_device[k_batch] = v_batch.to(model.device)
            elif isinstance(v_batch, list) and len(v_batch) > 0 and isinstance(v_batch[0], torch.Tensor):
                batch_device[k_batch] = [img.to(model.device) for img in v_batch]
            else:
                batch_device[k_batch] = v_batch
        
        batch = batch_device

        # 2. 从 LAPA 获取图像-问题联合特征 (Prophet的 'z')
        infer_results = model.infer(batch)
        img_ques_feats = infer_results["multi_modal_cls_feats"] 

        all_features.append(img_ques_feats.cpu())
        
        # 3. 收集 QIDs 和 Answers
        all_qids.extend(batch["qid"])
        if split == 'train': # <-- [新] 只在训练集时收集答案
            all_answers.extend(batch["vqa_answer"])

    return torch.cat(all_features, dim=0), all_qids, all_answers


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    dm = MTDataModule(_config, dist=False)
    dm.setup("fit") 

    model = M3AETransformerSS(_config)
    
    # ... (加载 checkpoint 的代码不变) ...
    if not _config["load_path"]:
        print("错误: 必须在配置中设置 `load_path` 来加载一个训练好的LAPA模型。")
        return
    print(f"正在从以下路径加载 checkpoint: {_config['load_path']}")
    ckpt = torch.load(_config['load_path'], map_location="cpu")
    if "state_dict" not in ckpt:
        print("警告: Checkpoint 文件中未找到 'state_dict'。假设ckpt文件本身就是 state_dict。")
        state_dict = ckpt
    else:
        state_dict = ckpt["state_dict"]
    if model.is_clip:
        state_dict = adapt_position_encoding(state_dict, after=_config['image_size'], patch_size=_config['patch_size'])
    else:
        state_dict = swin_adapt_position_encoding(state_dict, after=_config['image_size'])
    model.load_state_dict(state_dict, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    # --- (加载结束) ---

    dm_key = dm.dm_keys[0] # 获取数据集名称
    output_dir = os.path.join(_config["log_dir"], "example_jsons")
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 答案感知示例生成开始 ---
    
    # 1. 提取训练集所有特征 (作为查找库)
    train_features, train_qids, train_answers = extract_features(dm, model, "train")
    
    # 2. 提取测试集所有特征 (作为查询样本)
    test_features, test_qids, _ = extract_features(dm, model, "test") # 测试集答案为空
    
    print(f"训练集特征: {train_features.shape}, 训练 QIDs: {len(train_qids)}")
    print(f"测试集特征: {test_features.shape}, 测试 QIDs: {len(test_qids)}")
    
    # --- [新步骤: 保存训练集 QID -> GT Answers 映射] ---
    train_gt_map = {}
    for qid, ans_list in zip(train_qids, train_answers):
        train_gt_map[str(qid)] = ans_list
        
    map_filename = f"{output_dir}/{dm_key}_train_gt_map.json"
    with open(map_filename, 'w') as f:
        json.dump(train_gt_map, f, indent=4)
    print(f"已保存 训练集QID->答案 映射到 {map_filename}")
    # --- [修改结束] ---

    # 3. 计算相似度
    print("正在计算余弦相似度矩阵...")
    train_features = F.normalize(train_features.to(device), p=2, dim=1)
    test_features = F.normalize(test_features.to(device), p=2, dim=1)
    sim_matrix = torch.matmul(test_features, train_features.T)
    
    # 4. 获取 Top-E (E=20)
    E = 20 # Prophet 论文中使用了 N=16 或 N=20 [cite: 1282, 1461]
    print(f"正在查找 Top-{E} 相似示例...")
    top_k_indices = torch.topk(sim_matrix, E, dim=1).indices.cpu().tolist()

    # 5. 构建 QID 映射字典
    similar_qids_map = {}
    for i in tqdm(range(len(test_qids)), desc="构建 QID 映射"):
        test_qid = str(test_qids[i]) 
        train_indices_for_this_test = top_k_indices[i]
        similar_train_qids = [str(train_qids[idx]) for idx in train_indices_for_this_test]
        similar_qids_map[test_qid] = similar_train_qids

    # 6. 保存到文件 (应用显式排序)
    print(f"对 {len(similar_qids_map)} 个条目进行显式排序...")
    sorted_qids = sorted(similar_qids_map.keys(), key=int)
    sorted_examples = OrderedDict([(qid, similar_qids_map[qid]) for qid in sorted_qids])
    
    # ... (文件名逻辑不变) ...
    if dm_key == "vqa_vqa_rad":
        output_filename = f"{output_dir}/radvqa_examples.json"
    elif dm_key == "vqa_slack":
        output_filename = f"{output_dir}/slakevqa_examples.json"
    elif dm_key == "vqa_medvqa_2019":
        output_filename = f"{output_dir}/medvqa2019_examples.json"
    else:
        output_filename = f"{output_dir}/{dm_key}_examples.json"
         
    with open(output_filename, 'w') as f:
        json.dump(sorted_examples, f, indent=4)
    
    print(f"成功保存 {len(sorted_examples)} 个测试问题的相似示例到 {output_filename}")
    print("答案感知示例生成完毕。")