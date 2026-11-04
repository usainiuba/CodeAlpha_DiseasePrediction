"""Readable names, short explanations and the form layout."""

TITLES = {
    "radius": "Radius",
    "texture": "Texture",
    "perimeter": "Perimeter",
    "area": "Area",
    "smoothness": "Smoothness",
    "compactness": "Compactness",
    "concavity": "Concavity",
    "concave_points": "Concave points",
    "symmetry": "Symmetry",
    "fractal_dimension": "Fractal dimension",
}

HELP = {
    "radius": "Average distance from the centre of a cell nucleus to its edge.",
    "texture": "How much the grey shades vary inside the nucleus.",
    "perimeter": "Length of the outline of the nucleus.",
    "area": "Size of the nucleus.",
    "smoothness": "How much the radius varies from point to point. Lower is smoother.",
    "compactness": "How far the shape is from a neat circle. Higher is less round.",
    "concavity": "How deep the dents in the outline are.",
    "concave_points": "How many dents the outline has.",
    "symmetry": "How symmetrical the nucleus is.",
    "fractal_dimension": "How rough or wiggly the outline is.",
}

FIELD_GROUPS = [
    ("Size", ["radius", "perimeter", "area"]),
    ("Texture and surface", ["texture", "smoothness"]),
    ("Shape", ["compactness", "concavity", "concave_points", "symmetry", "fractal_dimension"]),
]


def title(name):
    return TITLES.get(name, name.replace("_", " ").capitalize())
