def calculateIntersection(a0, a1, b0, b1):
    if a0 >= b0 and a1 <= b1: # Contained
        intersection = a1 - a0
    elif a0 < b0 and a1 > b1: # Contains
        intersection = b1 - b0
    elif a0 < b0 and a1 > b0: # Intersects right
        intersection = a1 - b0
    elif a1 > b1 and a0 < b1: # Intersects left
        intersection = b1 - a0
    else: # No intersection (either side)
        intersection = 0

    return intersection

X0, Y0, X1, Y1, = [0, 0, 10, 10]
AREA = float((X1 - X0) * (Y1 - Y0))

# Rectangles to check
rectangles = [[15, 0, 20, 10], [0, 15, 10, 20], [0, 0, 5, 5], [0, 0, 5, 10], [0, 5, 10, 100], [0, 0, 100, 100]]

# Intersecting rectangles:
intersecting = []

for x0, y0, x1, y1 in rectangles:       
    width = calculateIntersection(x0, x1, X0, X1)        
    height = calculateIntersection(y0, y1, Y0, Y1)        
    area = width * height
    percent = area / AREA

    if (percent >= 0.5):
        intersecting.append([x0, y0, x1, y1])

print(intersecting)