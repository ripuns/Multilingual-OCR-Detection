def group_text(boxes):
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: b[1])
    grouped = []
    current = list(boxes[0])

    avg_height = sum([b[3] - b[1] for b in boxes]) / len(boxes)
    v_tol = avg_height * 0.5
    h_gap = avg_height * 1.5

    for box in boxes[1:]:
        x1, y1, x2, y2 = box

        if abs(y1 - current[1]) < v_tol and (x1 - current[2]) < h_gap:
            current[2] = max(current[2], x2)
            current[3] = max(current[3], y2)
        else:
            grouped.append(tuple(current))
            current = list(box)

    grouped.append(tuple(current))
    return grouped