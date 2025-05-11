def test_faithfulness(image_path, important_mask, threshold=0.2):
    """
    Run a faithfulness test on a model by masking out important regions.
    Returns a report string for overlay or logging.
    """
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Original prediction
    with torch.no_grad():
        orig_output = model(input_tensor)
        orig_prob = torch.softmax(orig_output, dim=1).squeeze()
        orig_score = orig_prob.max().item()
        orig_class = orig_prob.argmax().item()

    # Mask out the important region
    masked_tensor = input_tensor.clone()
    masked_tensor[:, :, important_mask] = 0.5  # Gray

    # Prediction after masking
    with torch.no_grad():
        masked_output = model(masked_tensor)
        masked_prob = torch.softmax(masked_output, dim=1).squeeze()
        masked_score = masked_prob[orig_class].item()

    drop = orig_score - masked_score
    faithful = "FAITHFUL ✅" if drop >= threshold else "NOT FAITHFUL ❌"

    report = (
        f"Original Score: {orig_score:.4f}\n"
        f"Masked Score: {masked_score:.4f}\n"
        f"Drop in Confidence: {drop:.4f}\n"
        f"Explanation is {faithful} (important region affected the prediction)"
    )
    return report
