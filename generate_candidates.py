import copy
import os
import resource
import json
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
from collections import OrderedDict 

from m3ae.config import ex
from m3ae.datamodules.multitask_datamodule import MTDataModule
from m3ae.modules import M3AETransformerSS

from m3ae.modules.vision_encoders.clip_model import adapt_position_encoding
from m3ae.modules.vision_encoders.swin_helpers import swin_adapt_position_encoding


# 设置资源限制
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def format_img_id(img_path, dataset_name):
    """
    根据数据集名称格式化 img_id
    """
    try:
        if "vqa_rad" in dataset_name:
            return os.path.splitext(os.path.basename(img_path))[0]
        elif "slack" in dataset_name:
            return os.path.basename(os.path.dirname(img_path))
        elif "medvqa_2019" in dataset_name:
            return os.path.splitext(os.path.basename(img_path))[0]
        else:
            return os.path.basename(img_path)
    except Exception:
        return img_path 

@torch.no_grad()
def generate_candidates_for_split(
    datamodule, model, id2answer, split='val', k_top=10, 
    answer2id=None, test_to_train_map=None, train_gt_map=None
):
    """
    为给定的数据划分（train, val, test）生成候选答案和概率。
    (k_top: 候选答案的数量)
    
    [MODIFIED]: 此版本将融合来自 '答案感知示例' 的答案，以提高召回率。
    """
    dm_key = datamodule.dm_keys[0] 
    
    dataloader = None
    if split == 'train':
        print(f"为 '{split}' 划分生成候选 (k={k_top}, shuffle=False, 融合邻居答案)。")
        # --- [修正] ---
        dataloader = torch.utils.data.DataLoader(
            datamodule.train_dataset,
            batch_size=datamodule.batch_size,
            sampler=None, 
            num_workers=datamodule.num_workers,
            collate_fn=datamodule.collate,
            shuffle=False 
        )
        # --- [修正结束] ---
    elif split == 'val':
        print(f"为 '{split}' 划分生成候选 (k={k_top}, shuffle=False, 融合邻居答案)。")
        dataloader = torch.utils.data.DataLoader(
            datamodule.val_dataset,
            batch_size=datamodule.batch_size,
            sampler=None,
            num_workers=datamodule.num_workers,
            collate_fn=datamodule.collate,
            shuffle=False 
        )
    else: # test
        print(f"为 '{split}' 划分生成候选 (k={k_top}, shuffle=False, 融合邻居答案)。")
        # --- [修正] ---
        dataloader = torch.utils.data.DataLoader(
            datamodule.test_dataset,
            batch_size=datamodule.batch_size,
            sampler=None,
            num_workers=datamodule.num_workers,
            collate_fn=datamodule.collate,
            shuffle=False 
        )
        # --- [修正结束] ---

    results = {}
    
    if not hasattr(model, 'vqa_head'):
        raise AttributeError("模型没有 'vqa_head'。请确保为VQA任务加载了正确的LAPA checkpoint。")

    for batch in tqdm(dataloader, desc=f"为 {split} 划分生成候选"):
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

        infer_results = model.infer(batch)
        img_ques_feats = infer_results["multi_modal_cls_feats"]

        # 1. 使用 LAPA 的 vqa_head 获取 *所有* logits
        y_pred_logits = model.vqa_head(img_ques_feats) 
        
        # 2. 获取 LaPA 模型的 Top-K 预测 (作为基础候选)
        top_k_logits, top_k_indices = torch.topk(y_pred_logits, k_top, dim=-1) 
        
        # 4. 格式化并存储结果
        batch_cpu = batch 
        batch_qids = batch_cpu["qid"]
        batch_img_ids_raw = batch_cpu["img_id"]
        batch_questions = batch_cpu["text"] 
        batch_ground_truth_list_of_lists = batch_cpu["vqa_answer"] 
        batch_answer_types = batch_cpu["answer_types"]

        top_k_indices_list = top_k_indices.cpu().numpy().tolist()
        
        for i in range(len(batch_qids)):
            qid = int(batch_qids[i])
            test_qid_str = str(qid)
            
            # --- [新步骤: 融合邻居答案] ---
            
            # 1. 获取 LaPA 模型的 Top-K 候选索引 (集合)
            model_candidate_indices = set(top_k_indices_list[i])
            
            # 2. 获取邻居的答案索引
            neighbor_candidate_indices = set()
            if test_to_train_map and train_gt_map and answer2id:
                # 2a. 找到相似的训练集 QIDs
                similar_train_qids = test_to_train_map.get(test_qid_str, [])
                
                # 2b. 查找这些 QID 的 GT 答案
                neighbor_answers_str = []
                for train_qid in similar_train_qids:
                    neighbor_answers_str.extend(train_gt_map.get(str(train_qid), []))
                
                # 2c. 将答案字符串转换为索引
                for ans_str in neighbor_answers_str:
                    if ans_str in answer2id:
                        neighbor_candidate_indices.add(answer2id[ans_str])

            # 3. 合并两个候选索引集
            combined_indices = model_candidate_indices.union(neighbor_candidate_indices)
            combined_indices_list = list(combined_indices)

            # 4. 从 *原始* logits 中提取这些合并后候选的 logits 和概率
            if not combined_indices_list: 
                combined_indices_list = top_k_indices_list[i] # 回退
            
            combined_indices_tensor = torch.tensor(combined_indices_list, dtype=torch.long, device=y_pred_logits.device) 
            
            combined_logits = y_pred_logits[i, combined_indices_tensor]
            
            # 5. 对这个 *扩充后* 的列表再次排序，选出新的 Top-K
            k_final = min(k_top, len(combined_indices_list))
            
            new_top_k_logits, sort_indices = torch.topk(combined_logits, k_final, dim=-1)
            
            # 映射回原始词汇表索引
            new_top_k_indices = [combined_indices_list[idx] for idx in sort_indices.cpu().numpy()]
            new_top_k_probs = torch.sigmoid(new_top_k_logits).cpu().numpy().tolist()

            # --- [修改结束] ---

            top_preds = [id2answer.get(idx, "unknown") for idx in new_top_k_indices]
            prediction = top_preds[0] 
            
            gt_ans_set = set(batch_ground_truth_list_of_lists[i])
            is_correct = prediction in gt_ans_set
                    
            atype_str = "CLOSED" if batch_answer_types[i] == 0 else "OPEN"
            
            formatted_img_id = format_img_id(batch_img_ids_raw[i], dm_key)
            
            results[qid] = {
                "img_id": formatted_img_id,
                "question": batch_questions[i],
                "answer": batch_ground_truth_list_of_lists[i][0] if batch_ground_truth_list_of_lists[i] else "N/A", 
                "top_k_preds": top_preds,
                "top_k_probs": [round(p, 6) for p in new_top_k_probs],
                "atype": atype_str, 
                "prediction": prediction,
                "result": is_correct
            }
    
    print(f"为 {split} 划分排序 {len(results)} 个条目...")
    try:
        sorted_qids = sorted(results.keys(), key=int)
        sorted_results = OrderedDict([(str(qid), results[qid]) for qid in sorted_qids]) 
        return sorted_results
    except Exception as e:
        print(f"排序时出错: {e}。返回未排序的结果。")
        return results


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    dm = MTDataModule(_config, dist=False)
    dm.setup("fit") 

    model = M3AETransformerSS(_config)
    
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
        state_dict = adapt_position_encoding(state_dict,
                                             after=_config['image_size'],
                                             patch_size=_config['patch_size'])
    else:
        state_dict = swin_adapt_position_encoding(state_dict, after=_config['image_size'])
        
    model.load_state_dict(state_dict, strict=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    dm_key = dm.dm_keys[0] 
    vqa_dm = dm.dm_dicts[dm_key]
    if not hasattr(vqa_dm, 'id2answer'):
        raise ValueError(f"Datamodule {dm_key} 未 setup。请确保 dm.setup('fit') 已被调用。")
        
    # --- [新步骤: 加载所有需要的映射文件] ---
    id2answer = vqa_dm.id2answer
    answer2id = vqa_dm.answer2id # <--- 新增
    print(f"从 {dm_key} 加载了 {len(id2answer)} 个唯一答案。")

    example_dir = os.path.join(_config["log_dir"], "example_jsons")
    
    # 构造文件路径
    if dm_key == "vqa_vqa_rad":
        example_filename = f"{example_dir}/radvqa_examples.json"
        gt_map_filename = f"{example_dir}/vqa_vqa_rad_train_gt_map.json"
    elif dm_key == "vqa_slack":
        example_filename = f"{example_dir}/slakevqa_examples.json"
        gt_map_filename = f"{example_dir}/vqa_slack_train_gt_map.json"
    elif dm_key == "vqa_medvqa_2019":
        example_filename = f"{example_dir}/medvqa2019_examples.json"
        gt_map_filename = f"{example_dir}/vqa_medvqa_2019_train_gt_map.json"
    else:
        example_filename = f"{example_dir}/{dm_key}_examples.json"
        gt_map_filename = f"{example_dir}/{dm_key}_train_gt_map.json"

    try:
        with open(example_filename, 'r') as f:
            test_to_train_map = json.load(f)
        print(f"成功加载相似邻居文件: {example_filename}")
    except FileNotFoundError:
        print(f"警告: 找不到相似邻居文件 {example_filename}。请先运行 generate_examples.py。")
        test_to_train_map = None

    try:
        with open(gt_map_filename, 'r') as f:
            train_gt_map = json.load(f)
        print(f"成功加载训练集GT映射文件: {gt_map_filename}")
    except FileNotFoundError:
        print(f"警告: 找不到训练集GT映射文件 {gt_map_filename}。请确保 generate_examples.py 已被正确修改并运行。")
        train_gt_map = None
    # --- [修改结束] ---

    
    K_FINAL = 10 
    
    for split in ["train", "val", "test"]:
        # 将所有映射传递下去
        candidates = generate_candidates_for_split(
            dm, model, id2answer, 
            split=split, 
            k_top=K_FINAL,
            answer2id=answer2id,
            test_to_train_map=test_to_train_map,
            train_gt_map=train_gt_map
        )
        
        output_dir = os.path.join(_config["log_dir"], "candidate_jsons_fused") # <-- 更改了输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        if dm_key == "vqa_vqa_rad" and split == "test":
             output_filename = f"{output_dir}/radvqa_results_fused.json"
        elif dm_key == "vqa_slack" and split == "test":
             output_filename = f"{output_dir}/slakevqa_results_fused.json"
        elif dm_key == "vqa_medvqa_2019" and split == "test":
             output_filename = f"{output_dir}/medvqa2019_results_fused.json"
        else:
             output_filename = f"{output_dir}/{dm_key}_{split}_candidates_fused.json"
        
        with open(output_filename, 'w') as f:
            json.dump(candidates, f, indent=4) 
        
        print(f"已保存 {len(candidates)} 个(融合邻居的)候选到 {output_filename}")

    print("所有(融合邻居的)候选生成完毕。")