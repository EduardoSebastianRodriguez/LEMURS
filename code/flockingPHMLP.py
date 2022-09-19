import torch
import argparse
from flockingFunctions import learnFlockingMLP, L2_loss, LD_loss, LH_loss, LV_loss


def main(numTrain=2000, numTests=400, numSamples=5, seed=42, numAgents=4):
    # Set fixed random number seed
    torch.manual_seed(seed)

    # Hyperparameters, the typical ones
    learning_rate   = 1e-3
    step_size       = 0.04
    time            = step_size * numSamples
    simulation_time = torch.linspace(0, time - step_size, numSamples)
    epochs          = 20000
    train_size      = 200
    tests_size      = 200
    alpha2          = 1.0
    alphaD          = 1.0
    alphaH          = 1.0
    alphaV          = 0.1

    # Parameters
    na     = torch.as_tensor(numAgents)  # Number of agents
    r      = torch.as_tensor(1.2*1.0) # Communication radius
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parameters = {"na": na, "device": device, "r": r}

    # Initialize the learned system
    learn_system = learnFlockingMLP(parameters)

    # Training mode
    learn_system.train()

    # Define the loss function and optimizer
    optimizer = torch.optim.Adam(learn_system.parameters(), lr=learning_rate, weight_decay=0.0)

    # Build dataset
    inputs = torch.load('flocking'+str(numAgents)+'_inputsTrain_' + str(numTrain) + str(numTests) + str(numSamples) + str(seed) + '_.pth').to(device)
    target_2 = torch.load('flocking'+str(numAgents)+'_target2Train_' + str(numTrain) + str(numTests) + str(numSamples) + str(seed) + '_.pth').to(device)

    # Build evaluation
    inputs_eval = torch.load('flocking'+str(numAgents)+'_inputsTests_' + str(numTrain) + str(numTests) + str(numSamples) + str(seed) + '_.pth').to(device)
    target_eval_2 = torch.load('flocking'+str(numAgents)+'_target2Tests_' + str(numTrain) + str(numTests) + str(numSamples) + str(seed) + '_.pth').to(device)
    current = 100000000000.0
    chosen  = torch.randperm(numTrain)[:train_size]

    # Log train and evaluation loss
    TrainLosses = []
    TestsLosses = []

    # Run the training loop
    for epoch in range(epochs):

        # Print epoch
        if epoch % 100 == 0:
            print(f'Starting epoch {epoch}')

        # Select batch
        if epoch % 100 == 0:
            chosen = torch.randperm(numTrain)[:train_size]

        # Run epoch
        outputs_2 = learn_system.forward(inputs[chosen, :], simulation_time, step_size)

        # Compute loss
        loss2 = L2_loss((outputs_2[:, :, :4 * na].reshape(-1, 4 * na)), target_2[:, chosen, :].reshape(-1, 4 * na))
        lossV = LV_loss((outputs_2[:, :, :4 * na].reshape(-1, 4 * na)))
        # lossD = LD_loss((outputs_2[:, :, :4 * na].reshape(-1, 4 * na)), target_2[:, chosen, :].reshape(-1, 4 * na), device)
        lossH = LH_loss(learn_system.dHdx(outputs_2.reshape(-1, 8 * na)))

        loss = alpha2 * loss2 + alphaH * lossH + alphaV * lossV #+ alphaD * lossD

        # Zero the gradients
        optimizer.zero_grad()

        # Perform backward pass
        loss.backward()

        # Perform optimization
        optimizer.step()

        # Print statistics
        if epoch % 10 == 0:
            print('Losses after epoch %5d: Loss = %.12f' % (epoch, loss2.detach().cpu().numpy()))

        # Validate
        if epoch % 500 == 0 or epoch == epochs - 1:

            chosen_eval = torch.randperm(numTests)[:tests_size]

            # Save train losses
            TrainLosses.append(loss.detach().cpu().numpy())

            # Print test
            print('-------------------------------------------------')

            # Perform forward pass
            learn_system.eval()

            # Run epoch
            outputs_2 = learn_system.forward(inputs_eval[chosen_eval, :], simulation_time, step_size)

            # Compute loss
            loss2 = L2_loss((outputs_2[:, :, :4 * na].reshape(-1, 4 * na)), target_eval_2[:, chosen_eval, :].reshape(-1, 4 * na))
            lossV = LV_loss((outputs_2[:, :, :4 * na].reshape(-1, 4 * na)))
            # lossD = LD_loss((outputs_2[:, :, :4 * na].reshape(-1, 4 * na)), target_eval_2[:, chosen_eval, :].reshape(-1, 4 * na), device)
            lossH = LH_loss(learn_system.dHdx(outputs_2.reshape(-1, 8 * na)))

            loss = alpha2 * loss2 + alphaH * lossH + alphaV * lossV #+ alphaD * lossD

            # Save tests losses
            TestsLosses.append(loss.detach().cpu().numpy())

            # Compute statistics
            print('Test results: Loss = %.6f' % loss2.detach().cpu().numpy())
            print('-------------------------------------------------')

            # Store model
            if current > loss.detach().cpu().numpy():
                torch.save(learn_system.state_dict(), 'flocking'+str(numAgents)+'_learn_system_phmlp.pth')
                current = loss.detach().cpu().numpy()

            # Go back to training mode
            learn_system.train()

    # Process is complete.
    print('Training process has finished.')

    # Save
    torch.save(TrainLosses, 'flocking'+str(numAgents)+'_learn_system_phmlp_TrainLosses.pth')
    torch.save(TestsLosses, 'flocking'+str(numAgents)+'_learn_system_phmlp_TestsLosses.pth')

    # Free unused memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='flocking')
    parser.add_argument('--numTrain', type=int, nargs=1, help='number of instances for training')
    parser.add_argument('--numTests', type=int, nargs=1, help='number of instances for testing')
    parser.add_argument('--numSamples', type=int, nargs=1, help='number of samples per instance')
    parser.add_argument('--seed', type=int, nargs=1, help='seed to generate reproducible data')
    parser.add_argument('--numAgents', type=int, nargs=1, help='number of agents')
    args = parser.parse_args()
    main(args.numTrain[0], args.numTests[0], args.numSamples[0], args.seed[0],  args.numAgents[0])
