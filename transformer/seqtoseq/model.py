import torch.nn.functional as F

class ReversePredictor(TransformerPredictor): 

    def _calculate_loss(self, batch, mode="train"): 
        inp_data, labels = batch 
        inp_data = F.one_hot(inp_data, num_classes=self.hparams.num_classes).float()

        preds = self.forward(inp_data, add_positional_encoding=True)
        # view(-1) will reshape labels to 1D 
        loss = F.cross_entropy(preds.view(-1, preds.size(-1)), labels.view(-1))
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log(f"{mode}_loss", loss)
        self.log(f"{mode}_acc", acc)
        return loss, acc 

    def training_step(self, batch, batch_idx): 
        loss, _ = self._calculate_loss(batch, mode="train")
        return loss 
    
    def validation_step(self, batch, batch_idx): 
        _ = self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx): 
        _ = self._calculate_loss(batch, mode="test")

    