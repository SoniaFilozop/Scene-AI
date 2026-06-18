import numpy as np

ALL_YOLO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


def extract_ontology_features(detected_objects):
    detected = {obj.lower() for obj in detected_objects}
    onto_features = {}

    cooccur_patterns = {
        "home_kitchen_cooking": {"refrigerator", "microwave", "oven", "sink"},
        "home_kitchen_utensils": {"fork", "knife", "spoon", "bowl", "cup"},
        "street_traffic": {"traffic light", "stop sign", "car", "bus", "truck"},
        "sports_equipment": {"sports ball", "tennis racket", "baseball bat", "skateboard"},
        "animals_nature": {"bird", "horse", "sheep", "cow", "bear", "zebra", "giraffe"},
        "home_living_room": {"couch", "chair", "bed", "dining table", "tv"}
    }
    for name, objs in cooccur_patterns.items():
        onto_features[f"cooccur_{name}"] = len(detected & objs) / len(objs) if objs else 0

    hierarchy = {
        "vehicle": {"car", "bus", "truck", "motorcycle", "bicycle", "airplane", "train", "boat"},
        "appliance": {"refrigerator", "oven", "microwave", "toaster", "sink"},
        "furniture": {"chair", "couch", "bed", "dining table", "bench"},
        "electronics": {"tv", "laptop", "cell phone", "keyboard", "mouse", "remote"},
        "utensil": {"fork", "knife", "spoon", "bowl", "cup", "wine glass"},
        "animal": {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"},
        "food": {"banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza"}
    }
    for cat, objs in hierarchy.items():
        onto_features[f"hier_{cat}"] = len(detected & objs)

    context_weights = {
        "street_context": {"traffic light": 3, "stop sign": 3, "car": 2, "bus": 2.5},
        "home_kitchen_context": {"refrigerator": 3, "oven": 3, "microwave": 3, "sink": 2.5},
        "sports_context": {"sports ball": 3, "tennis racket": 3, "skateboard": 2.5},
        "animals_context": {"bear": 4, "zebra": 4, "giraffe": 4, "elephant": 4},
        "home_furniture_context": {"couch": 2.5, "bed": 3, "toilet": 3, "dining table": 2}
    }
    for name, weights in context_weights.items():
        onto_features[f"context_{name}"] = sum(weights.get(obj, 0) for obj in detected)

    scene_rules = {
        "restaurant": {"wine glass", "cup", "fork", "knife", "spoon", "bowl", "pizza", "cake", "dining table"},
        "office": {"laptop", "keyboard", "mouse", "book"},
        "street": {"car", "bus", "truck", "motorcycle", "bicycle", "traffic light", "stop sign"},
        "store": {"handbag", "suitcase", "tie", "backpack"},
        "home": {"bed", "couch", "tv", "refrigerator", "oven", "microwave", "toaster", "sink"}
    }
    for scene, rules in scene_rules.items():
        onto_features[f"coverage_{scene}"] = len(detected & rules) / len(rules) if rules else 0

    key_objects = {
        "street": {"traffic light", "stop sign"},
        "restaurant": {"wine glass", "dining table"},
        "office": {"laptop", "keyboard"},
        "home": {"bed", "couch", "tv"},
        "store": {"handbag", "suitcase"}
    }
    for scene, keys in key_objects.items():
        onto_features[f"key_{scene}"] = 1 if (detected & keys) else 0

    onto_features["person_density"] = 1 if "person" in detected else 0
    onto_features["indoor_outdoor"] = 1 if any(x in detected for x in ["tv", "couch", "bed", "laptop"]) else 0

    return onto_features


def prepare_features(detected_objects):
    base_features = [1 if obj.lower() in {d.lower() for d in detected_objects} else 0 for obj in ALL_YOLO_CLASSES]
    onto_features = extract_ontology_features(detected_objects)
    combined = base_features + list(onto_features.values())

    return np.array([combined])
