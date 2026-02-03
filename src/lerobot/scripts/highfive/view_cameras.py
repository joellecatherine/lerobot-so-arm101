"""Quick script to visualize what both cameras see."""
import numpy as np
import cv2
from pathlib import Path
from lerobot.envs.highfive.highfive_env import HighFiveEnv


def main():
    # Create environment
    env = HighFiveEnv(
        render_mode="rgb_array",
        observation_width=480,
        observation_height=480,
        domain_randomization=False,
    )

    obs, info = env.reset()

    # Get images from both cameras
    birdseye = env._get_camera_image(camera_name="birdseye", width=480, height=480)
    wrist = env._get_camera_image(camera_name="wrist", width=480, height=480)

    # Convert RGB to BGR for OpenCV
    birdseye_bgr = cv2.cvtColor(birdseye, cv2.COLOR_RGB2BGR)
    wrist_bgr = cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR)

    # Add labels
    cv2.putText(birdseye_bgr, "BIRDSEYE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(wrist_bgr, "WRIST", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Stack side by side
    combined = np.hstack([birdseye_bgr, wrist_bgr])

    # Save images
    output_dir = Path(__file__).parent / "camera_views"
    output_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(output_dir / "birdseye.png"), birdseye_bgr)
    cv2.imwrite(str(output_dir / "wrist.png"), wrist_bgr)
    cv2.imwrite(str(output_dir / "combined.png"), combined)
    print(f"Saved camera views to: {output_dir}")

    # Show
    cv2.imshow("Camera Views (Press 's' to save, any other key to close)", combined)
    key = cv2.waitKey(0)

    if key == ord('s'):
        # Save with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(str(output_dir / f"combined_{timestamp}.png"), combined)
        print(f"Saved: combined_{timestamp}.png")

    cv2.destroyAllWindows()
    env.close()


if __name__ == "__main__":
    main()
