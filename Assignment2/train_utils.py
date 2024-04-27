import torch
import matplotlib.pyplot as plt

def train_gpt(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = []
    accuracy = []
    for inputs, masks, labels in train_loader:
        optimizer.zero_grad()
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
        outputs = model(inputs, masks)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        accuracy.append((torch.argmax(outputs.data, 1) == labels).sum().item() / len(labels))

    return sum(train_loss)/len(train_loss), sum(accuracy)/len(accuracy)

def train_kd(student, teacher, train_loader, optimizer, criterion_kl, criterion_ce, device, temperature, alpha):
    student.train()
    teacher.eval()
    train_loss = []
    accuracy = []
    for inputs, masks, labels in train_loader: 
        optimizer.zero_grad()
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device) 
        #print(inputs.shape)
        student_logits = student(inputs)
        with torch.no_grad():
            teacher_logits = teacher(inputs, masks)


        kl_loss = criterion_kl(torch.nn.functional.log_softmax(student_logits / temperature, dim=-1), torch.nn.functional.softmax(teacher_logits / temperature, dim=-1))
        ce_loss = criterion_ce(student_logits, labels)

        loss = alpha * kl_loss + (1 - alpha) * ce_loss

        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        accuracy.append((torch.argmax(student_logits.data, 1) == labels.to(device)).sum().item() / len(labels))

    return sum(train_loss)/len(train_loss), sum(accuracy)/len(accuracy)

def evaluate_gpt(model, val_loader, criterion, device):
    model.eval()
    val_loss = []
    accuracy = []
    with torch.no_grad():
        for inputs, masks, labels in val_loader:
            inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
            outputs = model(inputs, masks)
            loss = criterion(outputs, labels)
            val_loss.append(loss.item())
            accuracy.append((torch.argmax(outputs.data, 1) == labels).sum().item() / len(labels))
    return sum(val_loss)/len(val_loss), sum(accuracy)/len(accuracy)

def evaluate_kd(student, teacher, val_loader, criterion_kl, criterion_ce, device, temperature, alpha):
    student.eval()
    teacher.eval
    val_loss = []
    accuracy = []
    with torch.no_grad():
        for inputs, masks, labels in val_loader:
            inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
            teacher_logits = teacher(inputs, masks)
            student_logits = student(inputs)
            
            kl_loss = criterion_kl(torch.nn.functional.log_softmax(student_logits / temperature, dim=-1), torch.nn.functional.softmax(teacher_logits / temperature, dim=-1))
            ce_loss = criterion_ce(student_logits, labels)

            loss = alpha * kl_loss + (1 - alpha) * ce_loss 
            val_loss.append(loss.item())
            accuracy.append((torch.argmax(student_logits.data, 1) == labels).sum().item() / len(labels))
    return sum(val_loss)/len(val_loss), sum(accuracy)/len(accuracy)

def train_rnn(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = []
    accuracy = []
    for inputs, masks, labels in train_loader:
        optimizer.zero_grad()
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        accuracy.append((torch.argmax(outputs.data, 1) == labels).sum().item() / len(labels))

    return sum(train_loss)/len(train_loss), sum(accuracy)/len(accuracy)

def evaluate_rnn(model, val_loader, criterion, device):
    model.eval()
    val_loss = []
    accuracy = []
    with torch.no_grad():
        for inputs, masks, labels in val_loader:
            inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss.append(loss.item())
            accuracy.append((torch.argmax(outputs.data, 1) == labels).sum().item() / len(labels))
    return sum(val_loss)/len(val_loss), sum(accuracy)/len(accuracy)

def plot(train, val, title, path):
    plt.figure(figsize=(10,5))
    plt.title(title)
    plt.plot(val,label="val")
    plt.plot(train,label="train")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig(path)
