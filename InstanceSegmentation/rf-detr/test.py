from utils.load_dataset import SemanticSegDataset
if __name__ == "__main__":
    aa = SemanticSegDataset("/home/linde/Desktop/Datasets/2026-02-02-131248", (512, 512))
    ab = aa[2]
    print(len(ab))