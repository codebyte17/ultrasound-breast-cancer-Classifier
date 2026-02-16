
from src import get_device, data_loader ,load_yaml
import torch.nn as nn
from src import CustomCnnModel
from src import Trainer

def main():
    config = load_yaml("training_config.yaml")
    device = get_device()
    train_dataloader, test_dataloader = data_loader()
    model = CustomCnnModel()
    loss_fn = nn.CrossEntropyLoss()
    model_trainer = Trainer.from_config(config,
                                        model = model,
                                        device = device,
                                        train_loader=train_dataloader,
                                        test_loader=test_dataloader,
                                        loss_fn = loss_fn
                                       )

    model_trainer.train(epochs=config.epochs)

    model_trainer.plot_history()
    model_trainer.save_model()




if __name__ == "__main__":
    main()


'''
src/datasets (Data loaders)
src/metrics (we can use the evaluation matrix here like CE loss)
src/engine (epochs,)
src/models (Model architecture)
src/utils (get_device(),seed(), etc)
src/



'''