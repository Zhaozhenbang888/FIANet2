"""
Comprehensive Remote Sensing Vocabulary for Text Structure Parsing (GRL Module)

This module provides complete entity, attribute, and spatial relation lexicons
for three remote sensing datasets: NWPU-refer, RISBench_dataset, and RRSIS-D.

The lexicons are rule-based, deterministic, and require no training.
"""

# ============================================================================
# ENTITY LEXICON - Remote Sensing Target Objects
# ============================================================================

NWPU_ENTITIES = {
    # NWPU-refer categories from new_instances.json
    "car", "ship", "photovolatic", "building", "airplane", "road", "bridge",
    "basketball court", "container", "train", "pool", "storage tank", "tennis court",
    "ground track field", "football court", "pylon", "digger", "wind turbine",
    "construction tower", "road intersection", "parking", "baseball court", "river",
    "chimney", "airplaneroad", "ocean", "land", "rugby court", "lake", "grass",
    "badminton court", "dam", "no target",
    # aliases used by the NWPU text adapter for English prompts
    "solar panel", "photovoltaic panel", "stadium", "parking lot", "runway", "sea",
    "tower", "tower crane", "soccer field"
}

NWPU_ENTITY_ALIASES = {
    # common variants observed in prompts and annotations
    "ground track field", "ground-track-field", "track field", "groundtrackfield",
    "football court", "football-field", "soccer field",
    "baseball court", "baseball field", "baseballfield",
    "basketball court", "basketballcourt",
    "tennis court", "tenniscourt",
    "storage tank", "storage-tank", "storagetank",
    "road intersection", "intersection", "crossroad",
    "airplaneroad", "runway", "airstrip",
    "parking", "parking lot", "parking area",
    "wind turbine", "windturbine", "windmill",
    "construction tower", "tower crane", "crane",
}

RRSISD_ENTITIES = {
    # RRSIS-D dataset targets
    "airplane", "airport", "golf field", "expressway service area", "baseball field",
    "stadium", "ground track field", "storage tank", "basketball court", "chimney",
    "tennis court", "overpass", "train station", "ship", "expressway toll station",
    "dam", "harbor", "bridge", "vehicle", "windmill"
}

RRSISD_ENTITY_ALIASES = {
    # canonicalized from XML names in RRSIS-D/images/rrsisd/ann_split/*.xml
    "golffield", "golf field", "golf course",
    "trainstation", "train station",
    "storagetank", "storage tank", "storage-tank",
    "basketballcourt", "basketball court",
    "tenniscourt", "tennis court",
    "groundtrackfield", "ground track field", "ground-track-field",
    "baseballfield", "baseball field", "baseball court",
    "expressway-service-area", "expressway service area", "service area",
    "expressway-toll-station", "expressway toll station", "toll station",
}

REFSEGRS_ENTITIES = {
    # RefSegRS dataset targets (urban/road scene)
    "road", "vehicle", "car", "van", "building", "truck", "trailer", "bus",
    "road marking", "bikeway", "sidewalk", "tree", "low vegetation",
    "impervious surface"
}

RISBENCH_ENTITIES = {
    # RISBench dataset targets (comprehensive benchmark)
    # Land cover categories
    "water", "forest", "grassland", "cropland", "urban", "barren",
    # Building and infrastructure
    "building", "road", "railway", "airport", "harbor", "bridge",
    # Agricultural objects
    "farmland", "orchard", "vineyard",
    # Industrial objects
    "power plant", "factory", "chemical plant",
    # Transportation
    "airplane", "ship", "train", "vehicle", "bus", "truck", "car",
    # Sport facilities
    "stadium", "sports field", "tennis court", "basketball court", "baseball field",
    "golf course", "track field",
    # Public facilities
    "school", "hospital", "park", "swimming pool", "museum", "church",
    # Utilities
    "windmill", "windturbine", "dam", "reservoir", "storage tank", "tower",
    "transmission tower", "power tower", "communication tower",
    # Other
    "lake", "river", "mountain", "desert", "island", "glacier", "volcano"
}

RISBENCH_ENTITY_ALIASES = {
    # extracted from RISBench_dataset/output_phrase_*.txt surface forms
    "plane", "aircraft", "helicopter",
    "small vehicle", "large vehicle", "bus",
    "swimming pool", "pool",
    "ground-track-field", "ground track field",
    "soccer ball field", "soccer-ball-shaped field",
    "container crane", "harbor crane",
    "roundabout", "overpass",
}

ENTITY_LEXICON = (
    NWPU_ENTITIES
    | NWPU_ENTITY_ALIASES
    | RRSISD_ENTITIES
    | RRSISD_ENTITY_ALIASES
    | REFSEGRS_ENTITIES
    | RISBENCH_ENTITIES
    | RISBENCH_ENTITY_ALIASES
)


# ============================================================================
# ATTRIBUTE LEXICON - Remote Sensing Object Properties
# ============================================================================

SHAPE_ATTRIBUTES = {
    # Shape descriptors
    "elongated", "long", "thin", "narrow", "linear",          # 细长/狭长
    "circular", "round", "curved",                             # 圆形
    "rectangular", "square", "quadrilateral",                  # 长方形
    "polygonal", "irregular",
    "compact", "sprawling", "spread out",                      # 成片/蔓延
    "scattered", "sparse", "dispersed", "isolated",            # 零散/分散
}

COLOR_ATTRIBUTES = {
    # Color descriptors
    "white", "light", "bright",                                # 白色/浅色
    "red", "reddish", "dark red",                             # 红色
    "green", "dark green", "light green",                     # 绿色
    "blue", "dark blue",                                       # 蓝色
    "brown", "gray", "grey", "dark", "black",                 # 棕色/灰色
    "yellow", "orange", "pale", "light colored",
}

STATE_ATTRIBUTES = {
    # State/condition descriptors
    "dense", "densely packed", "concentrated",                 # 密集
    "sparse", "sparsely distributed",                          # 稀疏
    "fragmented", "broken", "discontinuous",                  # 碎片化
    "continuous", "connected", "coherent",                     # 连续
    "rotated", "tilted", "angled",                            # 旋转/倾斜
    "aligned", "parallel",
    "active", "operating", "functioning",                      # 运行中
    "abandoned", "unused", "idle",                            # 闲置/废弃
    "new", "old", "dilapidated",
    "regular", "well-maintained", "clean",
}

TEXTURE_ATTRIBUTES = {
    # Texture descriptors
    "smooth", "textured", "rough",
    "homogeneous", "uniform", "heterogeneous", "varied",
    "striped", "patterned", "arranged",
    "organized", "disorganized", "chaotic",
}

SIZE_ATTRIBUTES = {
    # very frequent in RISBench text phrases
    "large", "larger", "largest",
    "small", "smaller", "smallest",
    "big", "tiny", "huge",
}

POSITION_ATTRIBUTES = {
    # frequent ordinal/position descriptors from phrases
    "left-most", "right-most", "top-most", "bottom-most",
    "leftmost", "rightmost", "topmost", "bottommost",
    "middle", "center", "central",
    "upper", "lower",
}

ATTRIBUTE_LEXICON = (
    SHAPE_ATTRIBUTES
    | COLOR_ATTRIBUTES
    | STATE_ATTRIBUTES
    | TEXTURE_ATTRIBUTES
    | SIZE_ATTRIBUTES
    | POSITION_ATTRIBUTES
)


# ============================================================================
# SPATIAL RELATION LEXICON - Remote Sensing Scene Relationships
# ============================================================================

DIRECTIONAL_RELATIONS = {
    # Directional spatial relations
    "left", "left side", "left of",                            # 左侧
    "right", "right side", "right of",                         # 右侧
    "above", "up", "upper", "upper side", "on top",          # 上方
    "below", "down", "lower", "lower side", "under",         # 下方
    "north", "south", "east", "west",
    "northeast", "northwest", "southeast", "southwest",
}

PROXIMITY_RELATIONS = {
    # Proximity-based relations
    "near", "nearby", "close to",                              # 近/旁边
    "adjacent", "next to", "beside", "alongside",             # 相邻/旁边
    "far from", "distant from", "far away",
    "diagonal", "diagonally positioned",                       # 斜向
    "opposite", "across from",
}

TOPOLOGICAL_RELATIONS = {
    # Topological spatial relations
    "inside", "within", "in", "inside of",                     # 内部
    "outside", "outside of", "exterior",                       # 外部
    "surrounded by", "encircled by", "enclosed by",            # 被包围
    "surround", "encircle", "enclose",                         # 包围
    "between", "in between", "among",
    "around", "surrounding", "circumscribed",
    "overlapping", "overlapped by", "overlaps",
    "adjacent to", "touching", "contact with",
    "separate from", "separated",
}

HIERARCHICAL_RELATIONS = {
    # Part-whole or hierarchical relations
    "part of", "composed of", "containing", "contains",
    "portion of", "section of", "segment of",
    "connected to", "linked with", "attached to",
    "along", "along with", "following the line of",
    "parallel to", "perpendicular to", "intersecting",
}

ATTRIBUTE_RELATIONS = {
    # Relations based on attributes or categories
    "same as", "similar to", "resembles",
    "different from", "unlike",
    "larger than", "smaller than",
    "same size as", "different size",
    "aligned with", "misaligned with",
}

ACTION_RELATIONS = {
    # common verbs in RISBench expressions
    "parked", "docked", "positioned", "located",
    "spanning", "crossing", "extending",
    "surrounded by", "adjacent to", "closest to", "nearest to",
}

RELATION_LEXICON = (
    DIRECTIONAL_RELATIONS | PROXIMITY_RELATIONS | TOPOLOGICAL_RELATIONS |
    HIERARCHICAL_RELATIONS | ATTRIBUTE_RELATIONS | ACTION_RELATIONS
)


# ============================================================================
# CHINESE VOCABULARY (NWPU-refer)
# ============================================================================

NWPU_ENTITIES_ZH = {
    "汽车", "车辆", "船", "轮船", "舰船", "光伏", "光伏板", "太阳能板", "建筑", "楼房", "飞机", "道路", "公路", "马路",
    "桥", "篮球场", "集装箱", "火车", "泳池", "游泳池", "储罐", "油罐", "网球场", "田径场", "跑道", "足球场",
    "电塔", "铁塔", "挖掘机", "风力发电机", "风机", "施工塔", "塔吊", "路口", "道路交叉口", "停车场", "棒球场",
    "河流", "河", "烟囱", "海", "海洋", "陆地", "橄榄球场", "湖", "湖泊", "草地", "羽毛球场", "大坝"
}

NWPU_ATTRIBUTES_ZH = {
    "最大", "最小", "较大", "较小", "白色", "黑色", "红色", "蓝色", "黄色", "绿色", "灰色", "深色", "浅色",
    "矩形", "圆形", "密集", "稀疏", "连续", "分散"
}

NWPU_RELATIONS_ZH = {
    "左侧", "右侧", "左边", "右边", "上方", "下方", "上面", "下面", "左上角", "右上角", "左下角", "右下角",
    "中间", "附近", "旁边", "道路上", "海上", "河上", "停车场内", "跑道上", "屋顶", "在"
}

# Canonical maps consumed by GRL Chinese parser.
CHINESE_ENTITY_CANONICAL = {
    "汽车": "car", "车辆": "car",
    "船": "ship", "轮船": "ship", "舰船": "ship",
    "光伏": "photovolatic", "光伏板": "photovolatic", "太阳能板": "photovolatic",
    "建筑": "building", "楼房": "building",
    "飞机": "airplane",
    "道路": "road", "公路": "road", "马路": "road",
    "桥": "bridge",
    "篮球场": "basketball court",
    "集装箱": "container",
    "火车": "train",
    "泳池": "pool", "游泳池": "pool",
    "储罐": "storage tank", "油罐": "storage tank",
    "网球场": "tennis court",
    "田径场": "ground track field", "跑道": "ground track field",
    "足球场": "football court",
    "电塔": "pylon", "铁塔": "pylon",
    "挖掘机": "digger",
    "风力发电机": "wind turbine", "风机": "wind turbine",
    "施工塔": "construction tower", "塔吊": "construction tower",
    "路口": "road intersection", "道路交叉口": "road intersection",
    "停车场": "parking",
    "棒球场": "baseball court",
    "河流": "river", "河": "river",
    "烟囱": "chimney",
    "海": "ocean", "海洋": "ocean",
    "陆地": "land",
    "橄榄球场": "rugby court",
    "湖": "lake", "湖泊": "lake",
    "草地": "grass",
    "羽毛球场": "badminton court",
    "大坝": "dam",
}

CHINESE_ATTRIBUTE_CANONICAL = {
    "最大": "largest", "最小": "smallest", "较大": "larger", "较小": "smaller",
    "白色": "white", "黑色": "black", "红色": "red", "蓝色": "blue", "黄色": "yellow", "绿色": "green", "灰色": "gray",
    "深色": "dark", "浅色": "light", "矩形": "rectangular", "圆形": "circular",
    "密集": "dense", "稀疏": "sparse", "连续": "continuous", "分散": "dispersed",
}

CHINESE_RELATION_CANONICAL = {
    "左侧": ("left", "reference"), "右侧": ("right", "reference"),
    "左边": ("left", "reference"), "右边": ("right", "reference"),
    "上方": ("top", "reference"), "下方": ("bottom", "reference"),
    "上面": ("top", "reference"), "下面": ("bottom", "reference"),
    "左上角": ("top left", "reference"), "右上角": ("top right", "reference"),
    "左下角": ("bottom left", "reference"), "右下角": ("bottom right", "reference"),
    "中间": ("center", "reference"), "附近": ("near", "reference"), "旁边": ("near", "reference"),
    "道路上": ("on", "modification"), "海上": ("on", "modification"), "河上": ("on", "modification"),
    "停车场内": ("in", "modification"), "跑道上": ("on", "modification"), "屋顶": ("on", "modification"),
    "在": ("in", "modification"),
}


# ============================================================================
# DATASET-SPECIFIC LEXICON MAPPINGS
# ============================================================================

DATASET_LEXICONS = {
    "nwpu": {
        "entities": NWPU_ENTITIES | NWPU_ENTITY_ALIASES | NWPU_ENTITIES_ZH,
        "attributes": ATTRIBUTE_LEXICON | NWPU_ATTRIBUTES_ZH,
        "relations": RELATION_LEXICON | NWPU_RELATIONS_ZH,
        "entity_zh_map": CHINESE_ENTITY_CANONICAL,
        "attribute_zh_map": CHINESE_ATTRIBUTE_CANONICAL,
        "relation_zh_map": CHINESE_RELATION_CANONICAL,
    },
    "nwpu-refer": {
        "entities": NWPU_ENTITIES | NWPU_ENTITY_ALIASES | NWPU_ENTITIES_ZH,
        "attributes": ATTRIBUTE_LEXICON | NWPU_ATTRIBUTES_ZH,
        "relations": RELATION_LEXICON | NWPU_RELATIONS_ZH,
        "entity_zh_map": CHINESE_ENTITY_CANONICAL,
        "attribute_zh_map": CHINESE_ATTRIBUTE_CANONICAL,
        "relation_zh_map": CHINESE_RELATION_CANONICAL,
    },
    "nwpurefer": {
        "entities": NWPU_ENTITIES | NWPU_ENTITY_ALIASES | NWPU_ENTITIES_ZH,
        "attributes": ATTRIBUTE_LEXICON | NWPU_ATTRIBUTES_ZH,
        "relations": RELATION_LEXICON | NWPU_RELATIONS_ZH,
        "entity_zh_map": CHINESE_ENTITY_CANONICAL,
        "attribute_zh_map": CHINESE_ATTRIBUTE_CANONICAL,
        "relation_zh_map": CHINESE_RELATION_CANONICAL,
    },
    "rrsisd": {
        "entities": RRSISD_ENTITIES | RRSISD_ENTITY_ALIASES,
        "attributes": ATTRIBUTE_LEXICON,
        "relations": RELATION_LEXICON,
    },
    "refsegrs": {
        "entities": REFSEGRS_ENTITIES,
        "attributes": ATTRIBUTE_LEXICON,
        "relations": RELATION_LEXICON,
    },
    "risbench": {
        "entities": RISBENCH_ENTITIES | RISBENCH_ENTITY_ALIASES,
        "attributes": ATTRIBUTE_LEXICON,
        "relations": RELATION_LEXICON,
    },
    "rsibench_dataset": {
        "entities": RISBENCH_ENTITIES | RISBENCH_ENTITY_ALIASES,
        "attributes": ATTRIBUTE_LEXICON,
        "relations": RELATION_LEXICON,
    },
    "rsibench-dataset": {
        "entities": RISBENCH_ENTITIES | RISBENCH_ENTITY_ALIASES,
        "attributes": ATTRIBUTE_LEXICON,
        "relations": RELATION_LEXICON,
    },
    "rsibenchdataset": {
        "entities": RISBENCH_ENTITIES | RISBENCH_ENTITY_ALIASES,
        "attributes": ATTRIBUTE_LEXICON,
        "relations": RELATION_LEXICON,
    },
}

# Default: use unified lexicon for all datasets
DEFAULT_LEXICON = {
    "entities": ENTITY_LEXICON | NWPU_ENTITIES_ZH,
    "attributes": ATTRIBUTE_LEXICON | NWPU_ATTRIBUTES_ZH,
    "relations": RELATION_LEXICON | NWPU_RELATIONS_ZH,
    "entity_zh_map": CHINESE_ENTITY_CANONICAL,
    "attribute_zh_map": CHINESE_ATTRIBUTE_CANONICAL,
    "relation_zh_map": CHINESE_RELATION_CANONICAL,
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_lexicon_for_dataset(dataset_name: str = None) -> dict:
    """Get lexicon for a specific dataset or unified lexicon.
    
    Args:
        dataset_name: One of 'nwpu', 'rrsisd', 'refsegrs', 'risbench', or None for unified
        
    Returns:
        Dictionary with 'entities', 'attributes', 'relations' keys
    """
    if dataset_name is None:
        return DEFAULT_LEXICON
    dataset_name = dataset_name.lower().strip()
    return DATASET_LEXICONS.get(dataset_name, DEFAULT_LEXICON)


def get_all_entities() -> set:
    """Get union of all entities across all datasets."""
    return ENTITY_LEXICON


def get_all_attributes() -> set:
    """Get union of all attributes across all datasets."""
    return ATTRIBUTE_LEXICON


def get_all_relations() -> set:
    """Get union of all spatial relations across all datasets."""
    return RELATION_LEXICON


# ============================================================================
# STATISTICS
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Remote Sensing Text Structure Vocabulary Statistics")
    print("=" * 70)
    
    print("\n[ENTITIES]")
    print(f"  NWPU-refer:      {len(NWPU_ENTITIES)} entities")
    print(f"  RRSIS-D:         {len(RRSISD_ENTITIES)} entities")
    print(f"  RefSegRS:        {len(REFSEGRS_ENTITIES)} entities")
    print(f"  RISBench:        {len(RISBENCH_ENTITIES)} entities")
    print(f"  Total (unified): {len(ENTITY_LEXICON)} unique entities")
    
    print("\n[ATTRIBUTES]")
    print(f"  Shape:           {len(SHAPE_ATTRIBUTES)}")
    print(f"  Color:           {len(COLOR_ATTRIBUTES)}")
    print(f"  State:           {len(STATE_ATTRIBUTES)}")
    print(f"  Texture:         {len(TEXTURE_ATTRIBUTES)}")
    print(f"  Total:           {len(ATTRIBUTE_LEXICON)} unique attributes")
    
    print("\n[RELATIONS]")
    print(f"  Directional:     {len(DIRECTIONAL_RELATIONS)}")
    print(f"  Proximity:       {len(PROXIMITY_RELATIONS)}")
    print(f"  Topological:     {len(TOPOLOGICAL_RELATIONS)}")
    print(f"  Hierarchical:    {len(HIERARCHICAL_RELATIONS)}")
    print(f"  Attribute-based: {len(ATTRIBUTE_RELATIONS)}")
    print(f"  Total:           {len(RELATION_LEXICON)} unique relations")
    
    print("\n" + "=" * 70)
