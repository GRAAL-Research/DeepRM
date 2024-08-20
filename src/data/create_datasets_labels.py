import math


def create_datasets_labels(config: dict) -> list:
    if config["dataset"] == "moon":
        assert config["n_features"] == 2
        return ["-1", "+1"]

    elif config["dataset"] == "blob":
        assert config["n_features"] == 2
        return ["-1", "+1"]

    elif config["dataset"] in ["moon_and_blob", "blob_and_moon"]:
        assert config["n_features"] == 2
        return ["-1", "+1"]

    elif config["dataset"] == "mnist_binary":
        return ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"][
               :int((1 + math.sqrt(1 + 4 * int(config["n_dataset"]))) / 2)]

    elif config["dataset"] == "mnist_multi":
        return ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    elif config["dataset"] == "cifar100_binary":
        return ["beaver", "dolphin", "otter", "seal", "whale",
                "aquarium" "fish", "flatfish", "ray", "shark", "trout",
                "orchids", "poppies", "roses", "sunflowers", "tulips",
                "bottles", "bowls", "cans", "cups", "plates",
                "apples", "mushrooms", "oranges", "pears", "sweet peppers",
                "clock", "computer keyboard", "lamp", "telephone", "television",
                "bed", "chair", "couch", "table", "wardrobe",
                "bee", "beetle", "butterfly", "caterpillar", "cockroach",
                "bear", "leopard", "lion", "tiger", "wolf",
                "bridge", "castle", "house", "road", "skyscraper",
                "cloud", "forest", "mountain", "plain", "sea",
                "camel", "cattle", "chimpanzee", "elephant", "kangaroo",
                "fox", "porcupine", "possum", "raccoon", "skunk",
                "crab", "lobster", "snail", "spider", "worm",
                "baby", "boy", "girl", "man", "woman",
                "crocodile", "dinosaur", "lizard", "snake", "turtle",
                "hamster", "mouse", "rabbit", "shrew", "squirrel",
                "maple", "oak", "palm", "pine", "willow",
                "bicycle", "bus", "motorcycle", "pickup truck", "train",
                "lawn-mower", "rocket", "streetcar", "tank", "tractor"][:int((1 + math.sqrt(1 + 4 *
                                                                         int(config["n_dataset"]))) / 2)]

    elif config["dataset"] in ["MTPL2_frequency", "MTPL2_severity", "MTPL2_pure"]:
        return []

    raise NotImplementedError(f"The dataset '{config['dataset']}' is not supported.")
