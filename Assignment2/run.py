import argparse
from transformers import AutoTokenizer

from utils import *
from train_utils import *
from model import *
import copy

def main(args):
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    train_loader = get_data_loader(
        'data/in_domain_train.tsv', args.batch_size, tokenizer)
    val_loader = get_data_loader(
        'data/in_domain_dev.tsv', args.batch_size, tokenizer, shuffle=False)
    

    if args.mode == "gen":
        model = GPT(args.gpt_variant, is_gen=True).to(args.device)
        model.eval()

        # TODO: You can add your super creative prompt here
        prompt = "My name is Inigo Montoya. You killed my father. Prepare to die."

        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(args.device)
        output = model.generate(input_ids, max_new_tokens=args.max_new_tokens)
        print("", tokenizer.decode(output[0]), sep="\n")

    elif args.mode == "LoRA":    
        model = GPT(args.gpt_variant, LoRA_rank=args.LoRA_rank).to(args.device)
        
        # TODO: Implement the training loop (fill the train and evaluate functions in train_utils.py)
        # TODO: Also plot the training losses and metrics
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = torch.nn.CrossEntropyLoss()
        train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []
        best_acc = 0.0
        for epoch in range(args.epochs):
            print('EPOCH {}:'.format(epoch + 1))
            train_loss, train_accuracy = train_gpt(model, train_loader, optimizer, criterion, args.device)
            val_loss, val_accuracy = evaluate_gpt(model, val_loader, criterion, args.device)
            print('Train Loss: ', train_loss, 'Train Accuracy: ', train_accuracy)
            print('Val Loss: ', val_loss, 'Val Accuracy: ', val_accuracy)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            if val_accuracy > best_acc:
                best_model = copy.deepcopy(model)
                best_acc = val_accuracy

        best_model.save_trainable_params(args.model_path)
        print('Best validation accuracy = ', best_acc)
        plot(train_losses, val_losses, "Training and Validation Loss - LoRA", "plots/LoRA_loss.png")
        plot(train_accuracies, val_accuracies, "Training and Validation Accuracy - LoRA", "plots/LoRA_acc.png")
        
    elif args.mode == "distil":
        teacher_model = GPT(args.gpt_variant, LoRA_rank=args.LoRA_rank).to(args.device)
        teacher_model.load_trainable_params(args.model_path)
        teacher_model.eval()

        student_model = DistilRNN().to(args.device)  # TODO: Implement the student model class
        # TODO: Implement the training loop (fill the train and evaluate functions in train_utils.py)
        # HINT: You can use an additional parameter in train function to differentiate LoRA and distillation training, no changes in evaluate function required.
        optimizer = torch.optim.Adam(student_model.parameters(), lr=args.lr)
        criterion_ce = torch.nn.CrossEntropyLoss()
        criterion_kl = torch.nn.KLDivLoss()
        train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []
        best_acc = 0.0
        for epoch in range(args.epochs):
            print('EPOCH {}:'.format(epoch + 1))
            train_loss, train_accuracy = train_kd(student_model, teacher_model, train_loader, optimizer, criterion_kl, criterion_ce, args.device, temperature=3, alpha=0.75)
            val_loss, val_accuracy = evaluate_kd(student_model, teacher_model, val_loader, criterion_kl, criterion_ce, args.device, temperature=3, alpha=0.75)
            print('Train Loss: ', train_loss, 'Train Accuracy: ', train_accuracy)
            print('Val Loss: ', val_loss, 'Val Accuracy: ', val_accuracy)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            if val_accuracy > best_acc:
                best_acc = val_accuracy

        print('Best validation accuracy = ', best_acc)
        plot(train_losses, val_losses, "Training and Validation Loss - Knowledge Distillation", "plots/KD_loss.png")
        plot(train_accuracies, val_accuracies, "Training and Validation Accuracy - Knowledge Distillation", "plots/KD_acc.png")

    elif args.mode == "rnn":
        model = DistilRNN().to(args.device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = torch.nn.CrossEntropyLoss()
        train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []
        best_acc = 0.0
        for epoch in range(args.epochs):
            print('EPOCH {}:'.format(epoch + 1))
            train_loss, train_accuracy = train_rnn(model, train_loader, optimizer, criterion, args.device)
            val_loss, val_accuracy = evaluate_rnn(model, val_loader, criterion, args.device)
            print('Train Loss: ', train_loss, 'Train Accuracy: ', train_accuracy)
            print('Val Loss: ', val_loss, 'Val Accuracy: ', val_accuracy)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            if val_accuracy > best_acc:
                best_acc = val_accuracy

        print('Best validation accuracy = ', best_acc)
        plot(train_losses, val_losses, "Training and Validation Loss - RNN", "plots/RNN_loss.png")
        plot(train_accuracies, val_accuracies, "Training and Validation Accuracy - RNN", "plots/RNN_acc.png")
    else:
        print("Invalid mode")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assignment 2")
    parser.add_argument("mode", type=str, choices=["gen", "LoRA", "distil", "rnn"], help="Mode to run the program in")
    parser.add_argument("sr_no", type=int, help="5 digit SR number")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--gpt_variant", type=str, default="gpt2", choices=["gpt2", "gpt2-medium"], help="Model to use")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--model_path", type=str, default="models/LoRA.pth", help="Path to save the model")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--LoRA_rank", type=int, default=4, help="Low rank matrix bottleneck")
    # TODO: Add more arguments as needed
    
    args = parser.parse_args()
    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and args.gpu_id >= 0 else\
        "mps" if torch.backends.mps.is_available() else "cpu")
    
    seed_everything(args.sr_no)

    main(args)
