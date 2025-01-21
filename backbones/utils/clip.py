def convert_model_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()


def refine_classname(class_names):
    for i, class_name in enumerate(class_names):
        new_name = class_name.lower().replace("_", " ").replace("-", " ")
        new_name = (
            "an " + new_name
            if new_name[0] in ["a", "e", "i", "o", "u"]
            else "a " + new_name
        )
        class_names[i] = new_name
    return class_names
