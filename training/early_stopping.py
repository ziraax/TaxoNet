import numpy as np
import torch


class EarlyStopping:
    """
    Early-stops training when **either**
        • val_loss keeps getting worse  **or**
        • accuracy stops getting better
    for `patience` consecutive epochs.

    You can monitor one or both metrics; pass `monitor_acc=True/False`.
    """

    def __init__(
        self,
        patience: int = 10,
        delta: float = 0.0,
        path: str = "checkpoint.pt",
        monitor_acc: bool = True,
        verbose: bool = True,
        trace_func=print,
    ):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.monitor_acc = monitor_acc
        self.verbose = verbose
        self.trace_func = trace_func

        self.best_loss = np.inf
        self.best_acc = 0.0
        self.counter = 0
        self.early_stop = False

    # ------------------------------------------------------------------ #
    def __call__(self, val_loss: float, val_acc: float, model: torch.nn.Module):
        """
        Args
        ----
        val_loss : float
        val_acc  : float (ignored if monitor_acc=False)
        model    : torch.nn.Module
        """
        improved = False

        # --- loss criterion ------------------------------------------------ #
        if val_loss < self.best_loss - self.delta:
            improved = True
            self.best_loss = val_loss

        # --- accuracy criterion ------------------------------------------- #
        if self.monitor_acc and val_acc > self.best_acc + self.delta:
            improved = True
            self.best_acc = val_acc

        # ------------------------------------------------------------------ #
        if improved:
            # reset counter & save checkpoint
            self.counter = 0
            self._save_checkpoint(model, val_loss, val_acc)
        else:
            self.counter += 1
            self.trace_func(
                f"[EarlyStopping][INFO] No improvement for {self.counter}/{self.patience} epoch(s)"
            )
            if self.counter >= self.patience:
                self.early_stop = True

    # ------------------------------------------------------------------ #
    def _save_checkpoint(self, model: torch.nn.Module, val_loss: float, val_acc: float):
        if self.verbose:
            self.trace_func(
                f"[EarlyStopping][INFO] Saving new best : loss={val_loss:.4f}  acc={val_acc:.4f}"
            )
        torch.save(model.state_dict(), self.path)
