import numpy as np


def loadGloveModel(gloveFile, saveFile="glove_embeddings.npy"):
    try:
        print("Loading Glove Model")
        # Try to load preprocessed embeddings
        model = np.load(saveFile, allow_pickle=True).item()
        print("Done. {} words loaded from {}".format(len(model), saveFile))
        return model
    except FileNotFoundError:
        print("Preprocessed file not found. Loading and processing the original Glove file.")
        pass

    f = open(gloveFile, 'r', encoding="utf8")
    model = {}

    for line in f:
        try:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
        except Exception as e:
            # Handle exceptions if any
            print("Error processing line:", e)

    # Save the model to a file
    np.save(saveFile, model)
    print("Done. {} words loaded and saved to {}".format(len(model), saveFile))
    return model


glove_embeddings = loadGloveModel("C:/Users/ansar/Downloads/FYP Haris BUKC/Code Files GUI/flowers/glove.6B.300d.txt")
