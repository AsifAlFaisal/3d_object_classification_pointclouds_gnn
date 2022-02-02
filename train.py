#%%
from utils import *
from model import RotaInvNet
import torch
import optuna
from optuna.trial import TrialState
import mlflow
import numpy as np
# %%
# %%
def objective(trial):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EARLY_STOPPING_TOLERANCE = 10
    MAX_TRIAL = 100
    NUM_EPOCHS = 200
    MODEL_LIST = []
    with mlflow.start_run():
        if trial.number == MAX_TRIAL:
            print("Trial Stopped!")
            trial.study.stop()
        
        train_loader, test_loader, num_classes = get_data(batch_size=64)
        model = RotaInvNet(trial,in_dim=4, out_dim=num_classes).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=trial.suggest_float("lr", 1e-5, 1e-1, log=True))
        criterion = torch.nn.CrossEntropyLoss()

        min_loss = np.Inf
        NO_IMPROV = 0
        for epoch in range(NUM_EPOCHS):
            train_loss = train_one_epoch(model, optimizer, train_loader, criterion, DEVICE)

            with torch.no_grad():
                test_ba, test_acc, pred, test_loss = test_model(model, test_loader, criterion, DEVICE)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            trial.report(test_loss, epoch)
            if test_loss < min_loss:
                MODEL_LIST.append(model)
                NO_IMPROV = 0
                min_loss = test_loss
            else:
                NO_IMPROV += 1

            mlflow.log_metric("Train_Loss", train_loss, step=epoch)
            mlflow.log_metric("Test_Loss", test_loss, step=epoch)
            mlflow.log_metric("Balanced_Accuracy", test_ba, step=epoch)
            mlflow.log_metric("Accuracy", test_acc, step=epoch)
            mlflow.log_params(trial.params)
            if NO_IMPROV == EARLY_STOPPING_TOLERANCE:
                print(f'Early Stopping at epoch {epoch}')
                mlflow.log_metric('Early Stopping',epoch)
                break
            else:
                continue

        mlflow.pytorch.log_model(MODEL_LIST[-1], f"trl{trial.number}Model")
        return test_loss

# %%
if __name__=="__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    #
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")  