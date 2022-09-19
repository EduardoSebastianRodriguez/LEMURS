import torch
from navigationFunctions import learnNavigationGNN, L2_loss
import argparse

def main(numTrain=2000, numTests=400, numSamples=5, seed=42, numAgents=4):

    # Set fixed random number seed
    torch.manual_seed(seed)

    # Hyperparameters, the typical ones
    learning_rate   = 1e-3
    step_size       = 0.04
    time            = step_size * numSamples
    simulation_time = torch.linspace(0, time - step_size, numSamples)
    epochs          = 10000
    train_size      = 100
    tests_size      = 100

    # Parameters
    na     = torch.as_tensor(numAgents) # Number of agents
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Free unused memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    parameters = {"na": na, "device": device}

    # Initialize the learned system
    learn_system = learnNavigationGNN(parameters)

    # Training mode
    learn_system.train()

    # Define the loss function and optimizer
    optimizer   = torch.optim.Adam(learn_system.parameters(), lr=learning_rate)

    # Build dataset
    inputs   = torch.load('navigation'+str(numAgents)+'_inputsTrain_'+str(numTrain)+str(numTests)+str(numSamples)+str(42)+'_.pth').to(device)
    target_2 = torch.load('navigation'+str(numAgents)+'_target2Train_'+str(numTrain)+str(numTests)+str(numSamples)+str(42)+'_.pth').to(device)

    # Build evaluation
    inputs_eval   = torch.load('navigation'+str(numAgents)+'_inputsTests_'+str(numTrain)+str(numTests)+str(numSamples)+str(42)+'_.pth').to(device)
    target_eval_2 = torch.load('navigation'+str(numAgents)+'_target2Tests_'+str(numTrain)+str(numTests)+str(numSamples)+str(42)+'_.pth').to(device)
    current       = 100000000.0
    chosen        = torch.randperm(numTrain)[:train_size]

    # Log train and evaluation loss
    TrainLosses = []
    TestsLosses = []

    # Run the training loop
    for epoch in range(epochs):

        # Print epoch
        # if epoch % 100 == 0:
        #     print(f'Starting epoch {epoch}')

        # Select batch
        if epoch % 100 == 0:
            chosen = torch.randperm(numTrain)[:train_size]

        # Run epoch
        outputs_2 = learn_system.forward(inputs[chosen, :], simulation_time, step_size)

        # Compute loss
        loss = L2_loss((outputs_2[:, :, :4 * na].reshape(-1, 4 * na)), target_2[:, chosen, :].reshape(-1, 4 * na))

        # Zero the gradients
        optimizer.zero_grad()

        # Perform backward pass
        loss.backward()

        # Perform optimization
        optimizer.step()

        # Print statistics
        if epoch % 1000 == 0:
            print('Losses after epoch %5d: Loss = %.12f' % (epoch, loss.detach().cpu().numpy()))

        # Validate
        if epoch % 500 == 0 or epoch == epochs-1:

            chosen_eval = torch.randperm(numTests)[:tests_size]

            # Save train losses
            TrainLosses.append(loss.detach().cpu().numpy())

            # Print test
            # print('-------------------------------------------------')

            # Perform forward pass
            learn_system.eval()
            outputs_2 = learn_system.forward(inputs_eval[chosen_eval, :], simulation_time, step_size)

            # Compute loss
            loss = L2_loss((outputs_2[:, :, :4 * na].reshape(-1, 4 * na)), target_eval_2[:, chosen_eval, :].reshape(-1, 4 * na))

            # Save tests losses
            TestsLosses.append(loss.detach().cpu().numpy())

            # Compute statistics
            # print('Test results: Loss = %.12f' % (loss.detach().cpu().numpy()))
            # print('-------------------------------------------------')

            # Store model
            if current > loss.detach().cpu().numpy():
                torch.save(learn_system, 'navigation'+str(numAgents)+'_'+str(seed)+'_learn_system_phgnn.pth')
                current = loss.detach().cpu().numpy()

            # Go back to training mode
            learn_system.train()

    # Process is complete.
    print('Training process has finished.')

    # Save
    torch.save(TrainLosses, 'navigation'+str(numAgents)+'_'+str(seed)+'_learn_system_phgnn_TrainLosses.pth')
    torch.save(TestsLosses, 'navigation'+str(numAgents)+'_'+str(seed)+'_learn_system_phgnn_TestsLosses.pth')

    # Free unused memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Navigation problem, solved via the port-Hamiltonian GNN')
    parser.add_argument('--numTrain', type=int, nargs=1, help='number of navigation instances for training')
    parser.add_argument('--numTests', type=int, nargs=1, help='number of navigation instances for testing')
    parser.add_argument('--numSamples', type=int, nargs=1, help='number of samples per instance')
    parser.add_argument('--seed', type=int, nargs=1, help='seed to generate reproducible data')
    parser.add_argument('--numAgents', type=int, nargs=1, help='number of agents')
    args = parser.parse_args()
    main(args.numTrain[0], args.numTests[0], args.numSamples[0], args.seed[0],  args.numAgents[0])