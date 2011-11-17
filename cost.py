

def cost_a(k, w, b):
    # Shared aux
    return (3 * k + 4 * 2 * 32 * w + 2 * w * k) * 8 * b

def cost_b(k, w, b):
    # Non-shared aux
    return (3 * k * w + 4 * 2 * 32 * w) * 8 * b

def cost_c(k, w, b):
    # With reduction per 16 rows
    return (k * w + 4 * 2 * 32 * w) * 8 * b

for k, w, b in [
    (64, 2, 8),
    (32, 2, 8),
    ]:
    print (k, w, b), ":", cost_a(k, w, b), cost_b(k, w, b), cost_c(k, w, b)
