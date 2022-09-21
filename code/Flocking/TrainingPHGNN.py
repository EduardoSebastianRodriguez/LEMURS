import torch
import argparse
from Functions import learnSystemGNN, L2_loss


def main(numTrain=20000, numTests=20000, numSamples=5, seed_data=42, seed_train=42, numAgents=4):
    # Set fixed random number seed
    torch.manual_seed(seed_train)

    # Hyperparameters, the typical ones
    learning_rate   = 1e-3
    step_size       = 0.04
    time            = step_size * numSamples
    simulation_time = torch.linspace(0, time - step_size, numSamples)
    epochs          = 10000
    train_size      = 100
    tests_size      = 100

    # Parameters
    na     = torch.as_tensor(numAgents)  # Number of agents
    r      = torch.as_tensor(1.2*1.0) # Communication radius
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parameters = {"na": na, "device": device, "r": r}

    # Initialize the learned system
    learn_system = learnSystemGNN(parameters)

    # Training mode
    learn_system.train()

    # Define the loss function and optimizer
    optimizer = torch.optim.Adam(learn_system.parameters(), lr=learning_rate, weight_decay=0.0)

    # Build dataset
    inputs = torch.load('F'+str(numAgents)+'_inputsTrain_' + str(numTrain) + str(numTests) + str(numSamples) + str(seed_data) + '_.pth').to(device)
    target_2 = torch.load('F'+str(numAgents)+'_targetTrain_' + str(numTrain) + str(numTests) + str(numSamples) + str(seed_data) + '_.pth').to(device)

    # Build evaluation
    inputs_eval = torch.load('F'+str(numAgents)+'_inputsTests_' + str(numTrain) + str(numTests) + str(numSamples) + str(seed_data) + '_.pth').to(device)
    target_eval_2 = torch.load('F'+str(numAgents)+'_targetTests_' + str(numTrain) + str(numTests) + str(numSamples) + str(seed_data) + '_.pth').to(device)
    current = 1e25
    chosen  = torch.randperm(numTrain)[:train_size]

    # Log train and evaluation loss
    TrainLosses = []
    TestsLosses = []

    # Run the training loop
    for epoch in range(epochs):

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
        if epoch % 500 == 0 or epoch == epochs - 1:

            chosen_eval = torch.randperm(numTests)[:tests_size]

            # Perform forward pass
            learn_system.eval()

            # Run epoch
            outputs_2 = learn_system.forward(inputs_eval[chosen_eval, :], simulation_time, step_size)

            # Compute loss
            loss = L2_loss((outputs_2[:, :, :4 * na].reshape(-1, 4 * na)), target_eval_2[:, chosen_eval, :].reshape(-1, 4 * na))

            # Save tests losses
            TestsLosses.append(loss.detach().cpu().numpy())

            # Store model
            if current > loss.detach().cpu().numpy():
                torch.save(learn_system.state_dict(),  'F' + str(numAgents) + '_' + str(seed_train) + '_learn_system_phgnn.pth')
                current = loss.detach().cpu().numpy()
                TrainLosses.append(loss.detach().cpu().numpy())
            else:
                TrainLosses.append(TrainLosses[-1])

            # Go back to training mode
            learn_system.train()

    # Process is complete.
    print('Training process has finished.')

    # Save
    torch.save(TrainLosses, 'F'+str(numAgents)+'_'+str(seed_train)+'_learn_system_phgnn_TrainLosses.pth')
    torch.save(TestsLosses, 'F'+str(numAgents)+'_'+str(seed_train)+'_learn_system_phgnn_TestsLosses.pth')

    # Free unused memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GNN training for the flocking problem')
    parser.add_argument('--numTrain', type=int, nargs=1, help='number of instances for training')
    parser.add_argument('--numTests', type=int, nargs=1, help='number of instances for testing')
    parser.add_argument('--numSamples', type=int, nargs=1, help='number of samples per instance')
    parser.add_argument('--seed_data', type=int, nargs=1, help='seed used for generating the data')
    parser.add_argument('--seed_train', type=int, nargs=1, help='seed to generate reproducible data')
    parser.add_argument('--numAgents', type=int, nargs=1, help='number of agents')
    args = parser.parse_args()
    main(args.numTrain[0], args.numTests[0], args.numSamples[0], args.seed_data[0], args.seed_train[0],  args.numAgents[0])
