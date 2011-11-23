

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
    red = 3 * 2 * 32 * w * 8 * b
    acc = 2 * k * 8 * b

    cost = aux + red + acc
    
    print '%r aux=%d red=%d acc=%d =%d' % (
        (k, w, b), aux, red, acc, aux+red+acc)
#        cost_a(k, w, b), cost_b(k, w, b), cost_c(k, w, b))


print 'Precomputed data:'

nside = 2048
lmax = 2 * nside

per_col = 3 * 8 + 1 * 2
s = 0
for m in range(lmax + 1):
    for odd in range(2):
        s += 2 * nside * per_col
print s / 1024.**2

print 'map size', 12*nside**2 * 8 / 1024.**2
print 'alm size', (lmax + 1)**2 * 8 / 1024.**2
print 'q size', (2 * lmax + 1) * 16 * 2 * nside / 1024.**2
