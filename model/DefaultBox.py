"""
Defalut Box generator.

date: 9/30
author: arabian9ts
"""

def generate_boxes():
    """
    generate default boxes based on defined number
    the shape is [ [ first-feature-map-boxes ],
                   [ second-feature-map-boxes ],
                                ...
                   [ sixth-feature-map-boxes ], ]
        ==> ( 6 x defined_size )

    Returns: generated default boxes list
    """
    return