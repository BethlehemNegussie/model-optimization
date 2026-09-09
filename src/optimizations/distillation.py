"""
Use a large, accurate teacher model to train a smaller 
student model so the student is much smaller while 
retaining as much accuracy as possible."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F #imports PyTorch's functional API for math opns to calculate the distillation loss.
from torch.utils.data import DataLoader #DataLoader takes our dataset and gives the model batches
from tqdm import tqdm #progress bar you saw while training

#use @dataclass to create a configuration object containing the distillation settings
@dataclass
class DistillationConfig:
    temperature: float = 4.0   # tau: softens the teacher's distribution, controls how "soft" the teacher's predictions are, tries to expose hidden infor to the student
    alpha: float = 0.5         # weight on hard-label loss vs soft-target loss
    # hard and controls the balance between  soft losses, hard loss(uses actual ans),
    # soft loss(uses what teacher thinks), so stud learns both the ground truth and the teacher
    lr: float = 5e-5 #Learning rate to control how much the student's weights change during each optimization
    epochs: int = 3 #student sees the training dataset three times
    batch_size: int = 32 # student processes 32 examples at a time 


def distillation_loss(student_logits, teacher_logits, labels, temperature: float, alpha: float):
    """L_student = alpha * L_hard + (1 - alpha) * L_soft.
    calculates how badly the student is doing
    L_hard: standard cross-entropy against the ground-truth label.
    L_soft: KL divergence between temperature-softened student and teacher
    distributions, scaled by T^2 to keep gradient magnitudes
    comparable across temperature settings.
    """ 
    #receives  student's raw outputs, teacher's raw outputs, actual correct answers  
    
    #ordinary classification loss, tells us how wrong the student is      
    hard_loss = F.cross_entropy(student_logits, labels)
    #divide the teacher's logits by 4, makes the differences between classes less extreme
    # F.softmax converts those logits into probabilities
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    #log_softmax  used for student bc next operation is KL divergence, and PyTorch's F.kl_div() 
    # expects the input in log-probability form.  
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    #Compare teacher and student using KL divergence
    #T² compensates for distil loss becoming slower when we increase temperature so the gradients remain at a useful scale    
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature ** 2)
    #Combine the two losses into a single scalar loss value, weighted by alpha
    return alpha * hard_loss + (1 - alpha) * soft_loss, hard_loss.item(), soft_loss.item()

#actual student training loop
def train_distilled_student(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    train_dataset,
    config: DistillationConfig,
    input_keys=("input_ids", "attention_mask"),
    collate_fn=None,
    #If a GPU is available, use CUDA; otherwise use CPU.
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Teacher-guided distillation training loop.

    The teacher is frozen and run in eval/no_grad mode throughout -- only the
    student's parameters are updated. Works for either the NLP (BERT teacher
    -> smaller BERT/DistilBERT-style student) or vision (ViT teacher ->
    smaller ViT student) branch, since both consume the same {inputs, label}
    batch shape and produce `.logits`.
    """
    #.eval() Put teacher into evaluation mode(matters for layers such as dropout  ) 
    # .to(device) Move teacher onto the GPU so teach -> gpu.
    teacher.eval().to(device)
    #Student goes into training mode
    student.train().to(device)
    #optimize students params so it learns
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.lr)
    loader = DataLoader(
        #DataLoader turns dataset to baches
        train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn
    )

    history = [] #list used to store training statistics
    for epoch in range(config.epochs):
        epoch_loss, epoch_hard, epoch_soft, n_batches = 0.0, 0.0, 0.0, 0 #variables accumulate the losses throughout the epoch.
        for batch in tqdm(loader, desc=f"distill epoch {epoch + 1}/{config.epochs}"):
            #For every batch get batch -> teacher predicts -> student predicts -> calculate loss -> update student  
            #generic input prep
            #input_keys=("input_ids", "attention_mask")  for BERT/NLP inputs 
            # input_keys=("pixel_values",) for vit
            inputs = {k: batch[k].to(device) for k in input_keys if k in batch}
            #Get labels and moves the labels to the GPU  
            labels = (batch["labels"] if "labels" in batch else batch["label"]).to(device)

            with torch.no_grad(): #Don't calculate gradients for the teacher, only take the prediction
                teacher_logits = teacher(**inputs).logits
            #student receives the same input
            student_logits = student(**inputs).logits
            #Calculate distillation loss
            loss, hard, soft = distillation_loss(
                student_logits, teacher_logits, labels, config.temperature, config.alpha
            )
            #calculating the new gradient, clear gradients from the previous batch
            optimizer.zero_grad()
            loss.backward() #calculates gradients of the loss with respect to the student's parameters.
            optimizer.step() #AdamW uses those gradients to update the student's weights
            #accumulating statistics so we can calculate average losses at the end of the epoch
            epoch_loss += loss.item()
            epoch_hard += hard
            epoch_soft += soft
            n_batches += 1

        history.append({
            "epoch": epoch + 1,
            "loss": epoch_loss / n_batches,
            "hard_loss": epoch_hard / n_batches,
            "soft_loss": epoch_soft / n_batches,
        })
        print(history[-1])

    return student, history
