"""
Example usage of the AI Vision Assistant.
"""


def classification_example():
    """Example: Image Classification"""
    print("=" * 50)
    print("Image Classification Example")
    print("=" * 50)
    
    print("\nUsage:")
    print("  from vision_assistant import VisionAssistant")
    print("  assistant = VisionAssistant()")
    print("  results = assistant.classify('image.jpg', top_k=3)")
    print("\nExpected output:")
    print("  [")
    print("    {'label': 'golden retriever', 'score': 0.95},")
    print("    {'label': 'dog', 'score': 0.89},")
    print("    {'label': 'labrador', 'score': 0.75}")
    print("  ]")


def detection_example():
    """Example: Object Detection"""
    print("\n" + "=" * 50)
    print("Object Detection Example")
    print("=" * 50)
    
    print("\nUsage:")
    print("  from vision_assistant import VisionAssistant")
    print("  assistant = VisionAssistant()")
    print("  results = assistant.detect('scene.jpg', threshold=0.9)")
    print("\nExpected output:")
    print("  [")
    print("    {")
    print("      'label': 'person',")
    print("      'score': 0.98,")
    print("      'box': [100.5, 50.3, 300.7, 400.2]  # [x_min, y_min, x_max, y_max]")
    print("    },")
    print("    {")
    print("      'label': 'car',")
    print("      'score': 0.95,")
    print("      'box': [450.1, 200.6, 800.9, 450.3]")
    print("    }")
    print("  ]")


def segmentation_example():
    """Example: Image Segmentation"""
    print("\n" + "=" * 50)
    print("Image Segmentation Example")
    print("=" * 50)
    
    print("\nUsage:")
    print("  from vision_assistant import VisionAssistant")
    print("  assistant = VisionAssistant()")
    print("  results = assistant.segment('scene.jpg', threshold=0.9)")
    print("\nExpected output:")
    print("  {")
    print("    'segments': [")
    print("      {'id': 0, 'label': 'sky', 'score': 0.99, 'area': 50000},")
    print("      {'id': 1, 'label': 'person', 'score': 0.98, 'area': 15000},")
    print("      {'id': 2, 'label': 'ground', 'score': 0.97, 'area': 30000}")
    print("    ],")
    print("    'segmentation_map': numpy.array([[0, 0, 0, ...], ...])  # 2D array")
    print("  }")


def analyze_all_example():
    """Example: Analyze with all tasks"""
    print("\n" + "=" * 50)
    print("Complete Analysis Example")
    print("=" * 50)
    
    print("\nUsage:")
    print("  from vision_assistant import VisionAssistant")
    print("  assistant = VisionAssistant()")
    print("  # Run all tasks at once")
    print("  results = assistant.analyze('image.jpg')")
    print("  print(results['classification'])")
    print("  print(results['detection'])")
    print("  print(results['segmentation'])")
    print("\n  # Or run specific tasks")
    print("  results = assistant.analyze('image.jpg', tasks=['classify', 'detect'])")


def custom_models_example():
    """Example: Using custom models"""
    print("\n" + "=" * 50)
    print("Custom Models Example")
    print("=" * 50)
    
    print("\nUsage:")
    print("  # Use different pre-trained models")
    print("  assistant = VisionAssistant(")
    print("      classification_model='microsoft/resnet-50',")
    print("      detection_model='facebook/detr-resnet-101',")
    print("  )")
    print("  results = assistant.classify('image.jpg')")


def main():
    """Run all examples"""
    print("\n")
    print("╔════════════════════════════════════════════════╗")
    print("║     AI Vision Assistant - Usage Examples      ║")
    print("╔════════════════════════════════════════════════╗")
    
    classification_example()
    detection_example()
    segmentation_example()
    analyze_all_example()
    custom_models_example()
    
    print("\n" + "=" * 50)
    print("Getting Started")
    print("=" * 50)
    print("\n1. Install dependencies:")
    print("   pip install -r requirements.txt")
    print("\n2. Import and use:")
    print("   from vision_assistant import VisionAssistant")
    print("   assistant = VisionAssistant()")
    print("   results = assistant.classify('your_image.jpg')")
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
