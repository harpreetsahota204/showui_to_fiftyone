import fiftyone as fo
import pandas as pd
import numpy as np
import os
import json

def convert_bbox_to_fiftyone(bbox):
    """
    Convert bounding box coordinates from [x1, y1, x2, y2] format to FiftyOne format [x, y, width, height].
    All coordinates should be normalized in range [0, 1].
    
    Args:
        bbox (numpy.ndarray): Array of coordinates in format [x1, y1, x2, y2]
        
    Returns:
        list: Coordinates in FiftyOne format [x, y, width, height]
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1  # Calculate width as difference between x-coordinates
    height = y2 - y1  # Calculate height as difference between y-coordinates
    return [x1, y1, width, height]

def create_fiftyone_dataset(df, base_dir):
    """
    Create a FiftyOne dataset from a pandas DataFrame containing UI element metadata.
    
    Args:
        df (pandas.DataFrame): DataFrame containing:
            - image_url: Path to image file
            - bbox: Array of bounding boxes in [x1, y1, x2, y2] format
            - type: Array of element types/labels
            - instruction: Array of text content
            - point: Array of keypoints in [x, y] format
        base_dir (str): Base directory path (/home/harpreet/workspace/show-ui)
        
    Returns:
        fiftyone.core.dataset.Dataset: FiftyOne dataset containing all samples with their
        associated detections and keypoints
    """
    # Initialize empty dataset
    dataset = fo.Dataset(name="ShowUI_Web", overwrite=True)
    
    # Initialize list to store all samples
    samples = []
    
    # Define the correct images directory
    images_dir = os.path.join(base_dir, "ShowUI-web", "images")
    
    # Process each row (image) in the DataFrame
    for idx, row in df.iterrows():
        # Construct full path to image
        image_path = os.path.join(images_dir, row.image_url)
        
        # Create new sample with image
        sample = fo.Sample(filepath=image_path)

        sample[f"instructions"] = list(row.instruction)
        
        # Process bounding box detections
        detections = []
        for bbox, label, text in zip(row.bbox, row.type, row.instruction):
            # Create Detection object for each UI element
            detection = fo.Detection(
                label=label,  # Element type (e.g., Button, MenuItem)
                bounding_box=convert_bbox_to_fiftyone(bbox),  # Converted bbox coordinates
                text=text  # Element text content
            )
            detections.append(detection)
        
        # Add all detections to sample
        sample["detections"] = fo.Detections(detections=detections)
        
        # Process keypoints
        keypoints = []
        for point, label, text in zip(row.point, row.type, row.instruction):
            # Create Keypoint object for each UI element
            keypoint = fo.Keypoint(
                label=label,  # Element type (e.g., Button, MenuItem)
                points=[point],  # Keypoint coordinates (already in [x, y] format)
                text=text  # Element text content
            )
            keypoints.append(keypoint)
            
        # Add all keypoints to sample
        sample["keypoints"] = fo.Keypoints(keypoints=keypoints)
        
        # Add sample to list
        samples.append(sample)
    
    # Add all samples to dataset at once
    dataset.add_samples(samples)
    dataset.compute_metadata()
    dataset.add_dynamic_sample_fields()
    
    return dataset

def main():
    # Load the Parquet file into a pandas DataFrame
    df = pd.read_parquet('ShowUI-web/data/train-00000-of-00001.parquet')
    
    # Set base directory
    base_dir = "/home/harpreet/workspace/show-ui"
    
    # Create and process the dataset
    dataset = create_fiftyone_dataset(df, base_dir)
    
    # Optional: Launch the FiftyOne app to visualize the dataset
    # fo.launch_app(dataset)

if __name__ == "__main__":
    main()