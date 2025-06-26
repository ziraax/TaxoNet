def get_input_size_for_model(model_name: str, model_variant: str, config: dict = None) -> int:
    """
    Returns the input size for the specified model.

    Args:
        model_name (str): The name of the model.
        model_variant (str): The variant of the model (if applicable).
        config (dict, optional): If provided, img_size will be set inside it.

    Returns:
        int: The input size for the model.
    """
    input_size = 224

    if model_name.lower() == "efficientnet":
        if model_variant.lower() == "b0":
            input_size = 224
        elif model_variant.lower() == "b1":
            input_size = 240
        elif model_variant.lower() == "b2":
            input_size = 260
        elif model_variant.lower() == "b3":
            input_size = 300
        elif model_variant.lower() == "b4":
            input_size = 380
        elif model_variant.lower() == "b5":
            input_size = 456
        elif model_variant.lower() == "b6":
            input_size = 528
        elif model_variant.lower() == "b7":
            input_size = 600

    # Set in config only if provided
    if config is not None:
        config['img_size'] = input_size

    return input_size
