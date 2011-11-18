

def cost_a(k, w, b):
    # Shared aux, but with reduction buffer
    return (3 * k + 3 * 2 * 32 * w + 2 * w * k) * 8 * b + 2 * k * b

def cost_b(k, w, b):
    # Non-shared aux
    return (3 * k * w + 3 * 2 * 32 * w) * 8 * b + 2 * k * b

def cost_c(k, w, b):
    # Shared aux, with reduction per 16 rows
    return (3 * k + 3 * 2 * 32 * w + 2 * k) * 8 * b + 2 * k * b

print 'limit=', 48 * 1024

for k, w, b in [
    (64, 2, 8),
    (48, 2, 8),
    (32, 2, 8),
    ]:
    aux = 3 * k * 8 * b
    red = 4 * 2 * 32 * w * 8 * b
    acc = 2 * k * 8 * b
    
    print '%r aux=%d red=%d acc=%d %d %d %d' % (
        (k, w, b), aux, red, acc,
        cost_a(k, w, b), cost_b(k, w, b), cost_c(k, w, b))
