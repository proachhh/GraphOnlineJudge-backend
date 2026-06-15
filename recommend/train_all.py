import os
import sys
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def train_all(models='all', epochs_gnn=60, epochs_seq=30, epochs_rank=30,
              epochs_transe=150, epochs_rgcn=100):
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recommend_models')
    os.makedirs(model_dir, exist_ok=True)

    if models in ('all', 'gnn'):
        logger.info("=" * 60)
        logger.info("开始训练: GNN 图神经网络召回模型 (GraphSAGE)")
        logger.info("=" * 60)
        try:
            from recommend.gnn_recall import GNNTrainer
            trainer = GNNTrainer(save_dir=model_dir)
            trainer.export_graph_data()
            trainer.train(epochs=epochs_gnn, lr=0.001)
            logger.info("GNN 模型训练完成!")
        except Exception as e:
            logger.error(f"GNN 模型训练失败: {e}", exc_info=True)

    if models in ('all', 'sequence'):
        logger.info("=" * 60)
        logger.info("开始训练: 序列行为召回模型 (Transformer)")
        logger.info("=" * 60)
        try:
            from recommend.sequence_recall import SequenceRecallTrainer
            trainer = SequenceRecallTrainer(save_dir=model_dir, seq_len=20)
            trainer.export_sequence_data()
            trainer.train(epochs=epochs_seq, batch_size=64, lr=0.001)
            logger.info("序列召回模型训练完成!")
        except Exception as e:
            logger.error(f"序列召回模型训练失败: {e}", exc_info=True)

    if models in ('all', 'deepfm'):
        logger.info("=" * 60)
        logger.info("开始训练: DeepFM 精排模型")
        logger.info("=" * 60)
        try:
            from recommend.deepfm_ranking import DeepFMTrainer
            trainer = DeepFMTrainer(save_dir=model_dir)
            trainer.export_ranking_data(max_samples=50000)
            trainer.train_deepfm(epochs=epochs_rank, batch_size=256, lr=0.001, use_xdeepfm=False)
            logger.info("DeepFM 精排模型训练完成!")
        except Exception as e:
            logger.error(f"DeepFM 精排模型训练失败: {e}", exc_info=True)

    if models in ('all', 'xdeepfm'):
        logger.info("=" * 60)
        logger.info("开始训练: xDeepFM 精排模型")
        logger.info("=" * 60)
        try:
            from recommend.deepfm_ranking import DeepFMTrainer
            trainer = DeepFMTrainer(save_dir=model_dir)
            trainer.train_deepfm(epochs=epochs_rank, batch_size=256, lr=0.001, use_xdeepfm=True)
            logger.info("xDeepFM 精排模型训练完成!")
        except Exception as e:
            logger.error(f"xDeepFM 精排模型训练失败: {e}", exc_info=True)

    if models in ('all', 'simple'):
        logger.info("=" * 60)
        logger.info("开始训练: 旧版 MLP 模型 (SimpleRecommender) - fallback")
        logger.info("=" * 60)
        try:
            from recommend.train import train
            train()
            logger.info("旧版 MLP 模型训练完成!")
        except Exception as e:
            logger.error(f"旧版 MLP 模型训练失败: {e}", exc_info=True)

    if models in ('all', 'transe'):
        logger.info("=" * 60)
        logger.info("开始训练: TransE 知识图谱嵌入")
        logger.info("=" * 60)
        try:
            from knowledge_graph.kg_embedding import TransETrainer
            trainer = TransETrainer(save_dir=model_dir)
            trainer.train(epochs=epochs_transe, batch_size=512, lr=0.001, embed_dim=128)
            logger.info("TransE 嵌入训练完成!")
        except Exception as e:
            logger.error(f"TransE 训练失败: {e}", exc_info=True)

    if models in ('all', 'rgcn'):
        logger.info("=" * 60)
        logger.info("开始训练: RGCN 知识点关系推理")
        logger.info("=" * 60)
        try:
            from knowledge_graph.rgcn import RGCNTrainer
            trainer = RGCNTrainer(save_dir=model_dir)
            trainer.train(epochs=epochs_rgcn, lr=0.001)
            logger.info("RGCN 推理模型训练完成!")
        except Exception as e:
            logger.error(f"RGCN 训练失败: {e}", exc_info=True)

    logger.info("=" * 60)
    logger.info("所有模型训练完成!")
    logger.info(f"模型保存在: {model_dir}")
    logger.info("=" * 60)


def print_model_summary():
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recommend_models')
    legacy_dir = os.path.dirname(os.path.abspath(__file__))

    files = []

    if os.path.exists(os.path.join(model_dir, 'gnn_recall.pt')):
        files.append('GNN 图神经网络召回 (GraphSAGE)')
    if os.path.exists(os.path.join(model_dir, 'gnn_embeddings.pkl')):
        files.append('GNN 节点嵌入')
    if os.path.exists(os.path.join(model_dir, 'sequence_recall.pt')):
        files.append('序列行为召回 (Transformer)')
    if os.path.exists(os.path.join(model_dir, 'deepfm_ranking.pt')):
        files.append('DeepFM 精排模型')
    if os.path.exists(os.path.join(model_dir, 'xdeepfm_ranking.pt')):
        files.append('xDeepFM 精排模型')
    if os.path.exists(os.path.join(model_dir, 'graph_data.pkl')):
        files.append('图数据结构 (Neo4j导出)')
    if os.path.exists(os.path.join(model_dir, 'sequence_data.pkl')):
        files.append('序列数据结构')
    if os.path.exists(os.path.join(model_dir, 'ranking_data.pkl')):
        files.append('精排特征数据')
    if os.path.exists(os.path.join(legacy_dir, 'recommend_model.pt')):
        files.append('旧版 MLP 模型 (SimpleRecommender)')
    if os.path.exists(os.path.join(model_dir, 'transe.pt')):
        files.append('TransE 知识图谱嵌入')
    if os.path.exists(os.path.join(model_dir, 'kg_embeddings.pkl')):
        files.append('KG 实体嵌入向量')
    if os.path.exists(os.path.join(model_dir, 'rgcn.pt')):
        files.append('RGCN 知识点关系推理')
    if os.path.exists(os.path.join(model_dir, 'rgcn_embeddings.pkl')):
        files.append('RGCN 节点嵌入')
    if os.path.exists(os.path.join(model_dir, 'dqn_planner.pt')):
        files.append('DQN 强化学习路径规划')

    if files:
        print("已训练的模型:")
        for f in files:
            print(f"  - {f}")
    else:
        print("未找到已训练的模型，请运行: python recommender/train_all.py")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='多模态融合推荐系统训练脚本')
    parser.add_argument('--models', type=str, default='all',
                        choices=['all', 'gnn', 'sequence', 'deepfm', 'xdeepfm', 'simple',
                                 'transe', 'rgcn', 'dqn'],
                        help='指定要训练的模型 (默认: all)')
    parser.add_argument('--epochs-gnn', type=int, default=60, help='GNN 训练轮数')
    parser.add_argument('--epochs-seq', type=int, default=30, help='序列模型训练轮数')
    parser.add_argument('--epochs-rank', type=int, default=30, help='精排模型训练轮数')
    parser.add_argument('--epochs-transe', type=int, default=150, help='TransE 训练轮数')
    parser.add_argument('--epochs-rgcn', type=int, default=100, help='RGCN 训练轮数')
    parser.add_argument('--summary', action='store_true', help='显示已训练的模型摘要')

    args = parser.parse_args()

    if args.summary:
        print_model_summary()
    else:
        train_all(
            models=args.models,
            epochs_gnn=args.epochs_gnn,
            epochs_seq=args.epochs_seq,
            epochs_rank=args.epochs_rank,
            epochs_transe=args.epochs_transe,
            epochs_rgcn=args.epochs_rgcn,
        )
