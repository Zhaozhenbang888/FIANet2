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

RRSISD_ENTITIES = {
    # RRSIS-D dataset targets
    "airplane", "airport", "golf field", "expressway service area", "baseball field",
    "stadium", "ground track field", "storage tank", "basketball court", "chimney",
    "tennis court", "overpass", "train station", "ship", "expressway toll station",
    "dam", "harbor", "bridge", "vehicle", "windmill"
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

ENTITY_LEXICON = NWPU_ENTITIES | RRSISD_ENTITIES | REFSEGRS_ENTITIES | RISBENCH_ENTITIES


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

ATTRIBUTE_LEXICON = SHAPE_ATTRIBUTES | COLOR_ATTRIBUTES | STATE_ATTRIBUTES | TEXTURE_ATTRIBUTES


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

RELATION_LEXICON = (
    DIRECTIONAL_RELATIONS | PROXIMITY_RELATIONS | TOPOLOGICAL_RELATIONS |
    HIERARCHICAL_RELATIONS | ATTRIBUTE_RELATIONS
)


# ============================================================================
# DATASET-SPECIFIC LEXICON MAPPINGS
# ============================================================================

DATASET_LEXICONS = {
    "nwpu": {
        "entities": NWPU_ENTITIES,
        "attributes": ATTRIBUTE_LEXICON,
        "relations": RELATION_LEXICON,
    },
    "nwpu-refer": {
        "entities": NWPU_ENTITIES,
        "attributes": ATTRIBUTE_LEXICON,
        "relations": RELATION_LEXICON,
    },
    "nwpurefer": {
        "entities": NWPU_ENTITIES,
        "attributes": ATTRIBUTE_LEXICON,
        "relations": RELATION_LEXICON,
    },
    "rrsisd": {
        "entities": RRSISD_ENTITIES,
        "attributes": ATTRIBUTE_LEXICON,
        "relations": RELATION_LEXICON,
    },
    "refsegrs": {
        "entities": REFSEGRS_ENTITIES,
        "attributes": ATTRIBUTE_LEXICON,
        "relations": RELATION_LEXICON,
    },
    "risbench": {
        "entities": RISBENCH_ENTITIES,
        "attributes": ATTRIBUTE_LEXICON,
        "relations": RELATION_LEXICON,
    },
    "rsibench_dataset": {
        "entities": RISBENCH_ENTITIES,
        "attributes": ATTRIBUTE_LEXICON,
        "relations": RELATION_LEXICON,
    },
    "rsibench-dataset": {
        "entities": RISBENCH_ENTITIES,
        "attributes": ATTRIBUTE_LEXICON,
        "relations": RELATION_LEXICON,
    },
    "rsibenchdataset": {
        "entities": RISBENCH_ENTITIES,
        "attributes": ATTRIBUTE_LEXICON,
        "relations": RELATION_LEXICON,
    },
}

# Default: use unified lexicon for all datasets
DEFAULT_LEXICON = {
    "entities": ENTITY_LEXICON,
    "attributes": ATTRIBUTE_LEXICON,
    "relations": RELATION_LEXICON,
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
