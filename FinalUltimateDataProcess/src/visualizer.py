import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from src.utils import get_logger

logger = get_logger(__name__)


def visualize_3d_model(X, y, predictor=None):
    """
    Visualize 3D distribution of samples and the fitted model plane.

    Args:
        X: Feature matrix (speed, power)
        y: Target vector (usage_rate)
        predictor: Fitted Predictor instance (optional)
    """
    try:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        # X columns: 0 -> speed, 1 -> power
        xs = X[:, 0]
        ys = X[:, 1]
        zs = y

        # Scatter plot of data points
        scatter = ax.scatter(
            xs, ys, zs, c=zs, cmap="viridis", marker="o", alpha=0.6, label="Data Points"
        )
        fig.colorbar(scatter, ax=ax, label="Usage Rate")

        ax.set_xlabel("Speed Mean")
        ax.set_ylabel("Power Mean")
        ax.set_zlabel("Usage Rate (Accumulated Usage / Time)")

        if predictor is not None:
            # Create a meshgrid for the plane
            # We want to cover the range of speed and power
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()

            # Add some padding
            x_range = np.linspace(x_min, x_max, 20)
            y_range = np.linspace(y_min, y_max, 20)
            xx, yy = np.meshgrid(x_range, y_range)

            # Flatten to predict
            grid_X = np.c_[xx.ravel(), yy.ravel()]

            try:
                # Predict Z values for the grid
                pred_z = predictor.predict(grid_X)
                zz = pred_z.reshape(xx.shape)

                # Plot the surface
                ax.plot_surface(
                    xx,
                    yy,
                    zz,
                    alpha=0.3,
                    color="r",
                    rstride=1,
                    cstride=1,
                    edgecolor="none",
                )
                # Create a proxy artist for the legend
                import matplotlib.lines as mlines

                plane_proxy = mlines.Line2D(
                    [], [], color="r", alpha=0.3, label="Fitted Prediction Plane"
                )

                # Update legend
                handles, labels = ax.get_legend_handles_labels()
                handles.append(plane_proxy)
                ax.legend(handles=handles)

            except Exception as e:
                logger.warning(f"Could not plot fitted plane: {e}")

        plt.title("3D Distribution: Speed & Power vs Usage Rate")
        logger.info("Displaying 3D plot...")
        plt.show()

    except Exception as e:
        logger.error(f"Failed to visualize 3D model: {e}")


def save_2d_plot(X, y, save_path="plot_2d.png"):
    """
    Save a 2D scatter plot of Speed vs Power, colored by Usage Rate.

    Args:
        X: Feature matrix (speed, power)
        y: Target vector (usage_rate)
        predictor: Fitted Predictor instance (optional, currently unused for 2D plot but kept for consistency)
        save_path: Path to save the image
    """
    try:
        plt.figure(figsize=(10, 8))

        # X columns: 0 -> speed, 1 -> power
        xs = X[:, 0]
        ys = X[:, 1]
        zs = y

        scatter = plt.scatter(xs, ys, c=zs, cmap="viridis", alpha=0.6)
        plt.colorbar(scatter, label="Usage Rate")

        plt.xlabel("Speed Mean")
        plt.ylabel("Power Mean")
        plt.title("2D Distribution: Speed vs Power (Color: Usage Rate)")
        plt.grid(True, alpha=0.3)

        logger.info(f"Saving 2D plot to {save_path}...")
        plt.savefig(save_path)
        plt.close()

    except Exception as e:
        logger.error(f"Failed to save 2D plot: {e}")
