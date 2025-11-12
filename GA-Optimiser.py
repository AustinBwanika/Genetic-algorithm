# optimisers/ga_last_layer.py
import os, random, copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from deap import base, creator, tools
from tqdm import tqdm

from models.simple_cnn import SimpleCNN          # your CNN
from data.cifar10_loader import get_cifar10_loaders  # your loaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "experiments/baseline_model.pt"    # from your Adam baseline

# ------------------ Helpers to target ONLY the last layer ------------------ #
def freeze_all_but_last(model: SimpleCNN):
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.classifier[-1].parameters():
        p.requires_grad_(True)

def reinit_last_linear(model: SimpleCNN):
    last = model.classifier[-1]
    nn.init.kaiming_normal_(last.weight, nonlinearity="linear")
    if last.bias is not None:
        nn.init.zeros_(last.bias)

def get_last_layer_vector(model: SimpleCNN) -> torch.Tensor:
    last = model.classifier[-1]
    w = last.weight.detach().cpu().flatten()
    b = last.bias.detach().cpu() if last.bias is not None else torch.tensor([])
    return torch.cat([w, b])

def set_last_layer_vector(model: SimpleCNN, vec: torch.Tensor):
    last = model.classifier[-1]
    out_f, in_f = last.weight.shape
    w_num = out_f * in_f
    w = vec[:w_num].view(out_f, in_f).to(last.weight.device)
    last.weight.data.copy_(w)
    if last.bias is not None:
        b = vec[w_num:w_num+out_f].to(last.bias.device)
        last.bias.data.copy_(b)

@torch.no_grad()
def accuracy_on_loader(model: SimpleCNN, loader: DataLoader, max_batches=None):
    model.eval()
    total, correct, seen = 0, 0, 0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        preds = model(x).argmax(1)
        correct += (preds == y).sum().item()
        seen += y.size(0)
        total += 1
        if max_batches is not None and total >= max_batches:
            break
    return correct / seen

# ------------------ GA Fitness (validation accuracy) ------------------ #
def make_fitness_fn(model: SimpleCNN, val_loader: DataLoader, max_batches=50, clip=None):
    @torch.no_grad()
    def _fitness(individual):
        # individual is a Python list -> convert to tensor
        v = torch.tensor(individual, dtype=torch.float32)
        if clip is not None:
            v.clamp_(min=-clip, max=clip)
        set_last_layer_vector(model, v)
        return (accuracy_on_loader(model, val_loader, max_batches=max_batches),)
    return _fitness

# ------------------ GA main ------------------ #
def run_ga(
    pop_size=40,
    generations=50,
    cx_prob=0.7,
    mut_prob=0.3,
    tourn_size=3,
    init_sigma=0.05,
    mut_sigma=0.02,
    elitism=1,
    max_val_batches=50,
    seed=42,
):
    random.seed(seed)
    torch.manual_seed(seed)

    # 1) Load data (use your helper)
    train_loader, val_loader, test_loader = get_cifar10_loaders(batch_size=128)

    # 2) Build model and load baseline (pretrained full network)
    model = SimpleCNN(num_classes=10).to(DEVICE)
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Baseline checkpoint not found at {CHECKPOINT_PATH}. "
                                f"Run your Adam baseline first.")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))

    # 3) Freeze all but last layer; reinitialise last layer to make it fair
    freeze_all_but_last(model)
    reinit_last_linear(model)

    # 4) GA encoding: float vector = flattened (W,b) of last layer
    base_vec = get_last_layer_vector(model)
    dim = base_vec.numel()
    init_mean = 0.0
    clip = 0.25  # keep weights in a reasonable range

    # DEAP setup
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    tb = base.Toolbox()

    def init_gene():
        return random.gauss(init_mean, init_sigma)

    def init_individual():
        return creator.Individual([init_gene() for _ in range(dim)])

    tb.register("individual", init_individual)
    tb.register("population", tools.initRepeat, list, tb.individual)

    fitness_fn = make_fitness_fn(model, val_loader, max_batches=max_val_batches, clip=clip)
    tb.register("evaluate", fitness_fn)
    tb.register("mate", tools.cxBlend, alpha=0.25)                   # arithmetic crossover
    tb.register("mutate", tools.mutGaussian, mu=0.0, sigma=mut_sigma, indpb=0.5)
    tb.register("select", tools.selTournament, tournsize=tourn_size)

    # 5) GA loop
    pop = tb.population(n=pop_size)
    for ind in pop:
        ind.fitness.values = tb.evaluate(ind)

    best = tools.selBest(pop, 1)[0]
    print(f"Gen 0 | best val acc = {best.fitness.values[0]:.4f}")

    history = [best.fitness.values[0]]

    for gen in range(1, generations + 1):
        # Elitism + selection
        offspring = tools.selBest(pop, elitism) + tb.select(pop, len(pop) - elitism)
        offspring = list(map(tb.clone, offspring))

        # Crossover
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_prob:
                tb.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values

        # Mutation
        for ind in offspring:
            if random.random() < mut_prob:
                tb.mutate(ind)
                del ind.fitness.values

        # Re-evaluate
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = tb.evaluate(ind)

        pop[:] = offspring
        best = tools.selBest(pop, 1)[0]
        history.append(best.fitness.values[0])
        print(f"Gen {gen} | best val acc = {best.fitness.values[0]:.4f}")

    # 6) Set best weights, evaluate on full val & test, then save
    best_vec = torch.tensor(best, dtype=torch.float32)
    set_last_layer_vector(model, best_vec)
    full_val_acc = accuracy_on_loader(model, val_loader, max_batches=None)
    test_acc = accuracy_on_loader(model, test_loader, max_batches=None)
    print(f"Final VAL acc:  {full_val_acc:.4f}")
    print(f"Final TEST acc: {test_acc:.4f}")

    os.makedirs("experiments", exist_ok=True)
    torch.save(model.state_dict(), "experiments/ga_last_layer_model.pt")
    with open("experiments/ga_convergence.txt", "w") as f:
        for i, v in enumerate(history):
            f.write(f"{i},{v}\n")
    print("✅ GA-optimised model saved to experiments/ga_last_layer_model.pt")
    print("✅ Convergence saved to experiments/ga_convergence.txt")

if __name__ == "__main__":
    run_ga()
