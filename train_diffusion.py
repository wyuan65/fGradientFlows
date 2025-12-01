#!/usr/bin/env python3
"""
run_diffusion.py

Command-line interface for training and comparing diffusion flow and score models.
"""

import argparse
import torch
import diffusion_utils as du


def main():
    parser = argparse.ArgumentParser(description="Train and visualize diffusion flow/score models.")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for training")
    parser.add_argument("--device", type=str, default="cuda:7" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train_flow", action="store_true", help="Train the flow model")
    parser.add_argument("--train_equilibrium", action="store_true", help="Train the equilibrium model") 
    parser.add_argument("--train_vwgf", action="store_true", help="Train the vwgf model") 
    parser.add_argument("--train_score", action="store_true", help="Train the score model")
    parser.add_argument("--compare", action="store_true", help="Run comparison plots between models")
    parser.add_argument("--plot", action="store_true", help="Plot flow and score trajectories")
    parser.add_argument("--simulate", action="store_true", help="Simulate flow and score trajectories")

    args = parser.parse_args()
    device = torch.device(args.device)

    print(f"Using device: {device}")
    PARAMS = {
        "scale": 10.0,
        "target_scale": 5.0,
        "target_std": 1.0,
    }
    p_data = du.GaussianMixture.symmetric_2D(nmodes=5, std=PARAMS["target_std"], scale=PARAMS["target_scale"]).to(device)

    # Prepare conditional probability path
    path = du.GaussianConditionalProbabilityPath(
        p_data=p_data,
        alpha=du.LinearAlpha(),
        beta=du.SquareRootBeta(),
    ).to(device)

    # Shared model hyperparameters
    dim = 2
    hiddens = [16, 16, 16, 16]
    hiddens_critic = [16, 16, 16]

    flow_model, score_model, equilibrium_model, vwgf_model = None, None, None, None

    # -------------------------------
    # Train Flow Matching Model
    # -------------------------------
    if args.train_flow:
        print("\n=== Training Flow Matching Model ===")
        flow_model = du.MLPVectorField(dim=dim, hiddens=hiddens)
        flow_trainer = du.ConditionalFlowMatchingTrainer(path, flow_model)
        flow_trainer.train(
            num_epochs=args.epochs,
            device=device,
            lr=1e-3,
            batch_size=args.batch_size,
        )
        torch.save(flow_model.state_dict(), "flow_model.pt")
        print("Saved flow model to flow_model.pt")

    # -------------------------------
    # Train Equilibrium Matching Model
    # -------------------------------
    if args.train_equilibrium:
        print("\n=== Training Equilibrium Matching Model ===")
        equilibrium_model = du.MLPEquilibrium(dim=dim, hiddens=hiddens)
        equilibrium_trainer = du.ConditionalEquilibriumMatchingTrainer(path, equilibrium_model)
        equilibrium_trainer.train(
            num_epochs=args.epochs,
            device=device,
            lr=1e-3,
            batch_size=args.batch_size,
        )
        torch.save(equilibrium_model.state_dict(), "equilibrium_model.pt")
        print("Saved equilibrium model to equilibrium_model.pt")

     # -------------------------------
    # Train Variational Wasserstein Gradient Flow Model
    # -------------------------------
    if args.train_vwgf:
        print("\n=== Training Variational Wasserstein Gradient Flow Model ===")
        T = du.MLPTransport(dim=dim, hiddens=hiddens)
        h = du.MLPCritic(dim=dim, hiddens=hiddens_critic)
        P0 = du.Gaussian.isotropic(2, 1.0).to(device)
        Q = p_data
        #V = du.JSObjective(Q)
        #V = du.HellingerObjective(Q)
        #V = du.RKLObjective(Q)
        V = du.PearsonChiSquaredObjective(Q)
        #V = du.TVObjective(Q)
        a = 0.1
        vwgf_trainer = du.WGFTrainer(T, h, P0, Q, V, a)
        T_list = vwgf_trainer.train(
            K = 50,
            num_epochs = args.epochs,
            num_lambda_steps=1,
            num_theta_steps=1,
            device=device,
            batch_size=args.batch_size,
            lr=1e-3,
        )
        torch.save([T.state_dict() for T in T_list], "vwgf_model.pt")
        print("Saved vwgf model to vwgf_model.pt")

    # -------------------------------
    # Train Score Matching Model
    # -------------------------------
    if args.train_score:
        print("\n=== Training Score Matching Model ===")
        score_model = du.MLPScore(dim=dim, hiddens=hiddens)
        score_trainer = du.ConditionalScoreMatchingTrainer(path, score_model)
        score_trainer.train(
            num_epochs=args.epochs,
            device=device,
            lr=1e-3,
            batch_size=args.batch_size,
        )
        torch.save(score_model.state_dict(), "score_model.pt")
        print("Saved score model to score_model.pt")

    # -------------------------------
    # Load models (if needed)
    # -------------------------------
    if flow_model is None:
        try:
            flow_model = du.MLPVectorField(dim=dim, hiddens=hiddens)
            flow_model.load_state_dict(torch.load("flow_model.pt"))
            flow_model = flow_model.to(device)
            print("Loaded pretrained flow_model.pt")
        except FileNotFoundError:
            pass

    if equilibrium_model is None:
        try:
            equilibrium_model = du.MLPScore(dim=dim, hiddens=hiddens)
            equilibrium_model.load_state_dict(torch.load("equilibrium_model.pt"))
            equilibrium_model = score_model.to(device)
            print("Loaded pretrained equilibrium_model.pt")
        except FileNotFoundError:
            pass

    if score_model is None:
        try:
            score_model = du.MLPScore(dim=dim, hiddens=hiddens)
            score_model.load_state_dict(torch.load("score_model.pt"))
            score_model = score_model.to(device)
            print("Loaded pretrained score_model.pt")
        except FileNotFoundError:
            pass
    
    if vwgf_model is None:
        try:
            checkpoint = torch.load("vwgf_model.pt", map_location=device)
            T_list_loaded = []
            for state in checkpoint:
                T_k = du.MLPTransport(dim=dim, hiddens=hiddens).to(device)
                T_k.load_state_dict(state)
                T_k.eval()
                T_list_loaded.append(T_k)
            vwgf_model = T_list_loaded
            print(f"Loaded {len(vwgf_model)} transport maps from vwgf_model.pt")
        except FileNotFoundError:
            pass
    # -------------------------------
    # Comparisons
    # -------------------------------
    #if args.compare and flow_model and score_model:
    #    print("\n=== Comparing Vector Fields and Scores ===")
    #    du.compare_vector_fields(path, flow_model, score_model)
    #    du.compare_scores(path, flow_model, score_model)

    if args.compare and flow_model and equilibrium_model:
        print("\n=== Comparing Vector Fields and Equilibrium Fields ===")
        du.compare_vector_equilibrium(path, flow_model, equilibrium_model) 

    # -------------------------------
    # Plot Trajectories
    # -------------------------------
    if args.plot:
        print("\n=== Plotting Flow and Score Simulations ===")
        if flow_model:
            pass
            #flow_score_model = du.ScoreFromVectorField(flow_model,path.alpha,path.beta)
            #du.plot_flow(path, flow_model, 1000, output_file="flow_trajectory.pdf")
            #du.plot_score(path, flow_model, flow_score_model, 300, output_file="flow_stochastic_trajectory.pdf")
        if equilibrium_model:
            pass
            #du.plot_equilibrium(path, equilibrium_model, 1000, output_file="equilibrium_trajectory.pdf")
        if score_model:
            pass
            #score_flow_model = du.VectorFieldFromScore(score_model,path.alpha,path.beta) 
            #du.plot_flow(path, score_flow_model, 1000, output_file="score_deterministic_trajectory.pdf") 
            #du.plot_score(path, score_flow_model, score_model, 300, output_file="score_trajectory.pdf")
            #du.plot_score(path, flow_model, score_model, 300, output_file="score_flow_trajectory.pdf")
        if vwgf_model:
            P0 = du.Gaussian.isotropic(2, 1.0).to(device)
            Q = p_data
            du.plot_vwgf(P0, Q, vwgf_model, output_file="vwgf_trajectory.pdf") 
    # -------------------------------
    # Simulate
    # -------------------------------
    if args.simulate:
        num_samples = 10000
        timestep_list = torch.tensor([8,16,32,64,128,256,512,1024])  # sweep values

        results = {}
        if flow_model:
            flow_score_model = du.ScoreFromVectorField(flow_model, path.alpha, path.beta)
            W_list = []
            for num_steps in timestep_list:
                samples = du.simulate_flow(path, flow_model, num_samples, num_steps)
                target_samples = path.p_data.sample_projected(num_samples)
                W = du.wasserstein_distance(samples, target_samples)
                W_list.append(W.cpu())
                print(f"[Flow] Steps={num_steps:<4d} W={W:.4f}")
            results["flow_deterministic"] = (timestep_list, W_list)

            W_list = []
            for num_steps in timestep_list:
                samples = du.simulate_score(path, flow_model, flow_score_model, num_samples, num_steps)
                target_samples = path.p_data.sample_projected(num_samples)
                W = du.wasserstein_distance(samples, target_samples)
                W_list.append(W.cpu())
                print(f"[Flow Stochastic] Steps={num_steps:<4d} W={W:.4f}")
            results["flow_stochastic"] = (timestep_list, W_list)
        if score_model:
            score_flow_model = du.VectorFieldFromScore(score_model, path.alpha, path.beta)
            W_list = []
            for num_steps in timestep_list:
                samples = du.simulate_score(path, score_flow_model, score_model, num_samples, num_steps)
                target_samples = path.p_data.sample_projected(num_samples)
                W = du.wasserstein_distance(samples, target_samples)
                W_list.append(W.cpu())
                print(f"[Score] Steps={num_steps:<4d} W={W:.4f}")
            results["score_stochastic"] = (timestep_list, W_list)

            W_list = []
            for num_steps in timestep_list:
                samples = du.simulate_flow(path, score_flow_model, num_samples, num_steps)
                target_samples = path.p_data.sample_projected(num_samples)
                W = du.wasserstein_distance(samples, target_samples)
                W_list.append(W.cpu())
                print(f"[Score Deterministic] Steps={num_steps:<4d} W={W:.4f}")
            results["score_deterministic"] = (timestep_list, W_list)
        du.plot_results(results)
if __name__ == "__main__":
    main()
