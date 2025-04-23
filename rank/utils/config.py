########################################################################
               #################加载配置文件##############
########################################################################
import yaml
from collections import defaultdict, namedtuple
from pathlib import Path
SparseFeat = namedtuple('SparseFeat', ['name', 'voc_size', 'hash_size', 'share_embed','embed_dim', 'dtype'])
DenseFeat = namedtuple('DenseFeat', ['name', 'pre_embed','reduce_type','dim', 'dtype'])
VarLenSparseFeat = namedtuple('VarLenSparseFeat',['name', 'voc_size', 'hash_size', 'share_embed', 'weight_name', 'combiner', 'embed_dim', 'maxlen', 'dtype'])

def load_configs(feature_path=f"{Path(__file__).resolve().parent.parent}/configs/feature_config.yaml", 
                 model_path=f"{Path(__file__).resolve().parent.parent}/configs/model_config.yaml", 
                 train_path=f"{Path(__file__).resolve().parent.parent}/configs/train_config.yaml"):
    """加载并合并所有配置文件"""
    config = defaultdict(dict)
    
    try:
        # 加载特征配置
        with open(feature_path) as f:
            feature_config = yaml.safe_load(f)
            config['feature'] = feature_config
        
        # 加载模型配置 
        with open(model_path) as f:
            model_config = yaml.safe_load(f)
            config['model'] = model_config
            
        # 加载训练配置
        with open(train_path) as f:
            train_config = yaml.safe_load(f)
            config['training'] = train_config
            
    except FileNotFoundError as e:
        raise ValueError(f"Config file not found: {str(e)}")
    except yaml.YAMLError as e:
        raise ValueError(f"YAML parsing error: {str(e)}")
    
    # 配置完整性检查
    _validate_config(config)
    
    return dict(config)  # 转换为普通字典

def _validate_config(config):
    """配置校验逻辑"""
    required_sections = ['feature', 'model', 'training']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
            
    # 特征配置校验示例
    if 'feature_types' not in config['feature']:
        raise ValueError("Missing feature_types in feature config")
    

if __name__ == "__main__":
    config = load_configs()
    print()
    for ft in config['feature']['bucket_config']:
        print(ft)