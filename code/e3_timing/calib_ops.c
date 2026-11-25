/* E3a - calibracao de t_add e t_mul em double no PC.
 *
 * Latencia: cadeia dependente (cada operacao espera a anterior), 8 operacoes
 * por iteracao para diluir o custo do laco. Vazao: 8 acumuladores
 * independentes. Sem -ffast-math, entao o compilador nao pode reassociar.
 * Repeticoes: REPS medicoes de NOPS operacoes, reporta mediana e IQR.
 * Afinidade fixa num nucleo e prioridade alta para reduzir ruido.
 *
 * Compilar: gcc -O2 -march=native -o calib_ops calib_ops.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

#define NOPS 200000000LL
#define REPS 31

static volatile double sink;
static volatile double vin_add = 1e-9, vin_mul = 1.0 + 1e-12, vin_x0 = 1.0;

static double now_s(void) {
    LARGE_INTEGER f, t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)f.QuadPart;
}

static int cmp(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double lat_add(void) {
    double x = vin_x0, c = vin_add;
    double t0 = now_s();
    for (long long i = 0; i < NOPS / 8; i++) {
        x += c; x += c; x += c; x += c; x += c; x += c; x += c; x += c;
    }
    double t1 = now_s();
    sink = x;
    return (t1 - t0) / (double)NOPS;
}

static double lat_mul(void) {
    double x = vin_x0, m = vin_mul;
    double t0 = now_s();
    for (long long i = 0; i < NOPS / 8; i++) {
        x *= m; x *= m; x *= m; x *= m; x *= m; x *= m; x *= m; x *= m;
    }
    double t1 = now_s();
    sink = x;
    return (t1 - t0) / (double)NOPS;
}

static double thr_add(void) {
    double c = vin_add, a0 = vin_x0, a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    double t0 = now_s();
    for (long long i = 0; i < NOPS / 8; i++) {
        a0 += c; a1 += c; a2 += c; a3 += c; a4 += c; a5 += c; a6 += c; a7 += c;
    }
    double t1 = now_s();
    sink = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    return (t1 - t0) / (double)NOPS;
}

static double thr_mul(void) {
    double m = vin_mul, a0 = vin_x0, a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    double t0 = now_s();
    for (long long i = 0; i < NOPS / 8; i++) {
        a0 *= m; a1 *= m; a2 *= m; a3 *= m; a4 *= m; a5 *= m; a6 *= m; a7 *= m;
    }
    double t1 = now_s();
    sink = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    return (t1 - t0) / (double)NOPS;
}

static void run(const char *name, double (*f)(void), double *med) {
    double v[REPS];
    f(); /* aquecimento */
    for (int r = 0; r < REPS; r++) v[r] = f();
    qsort(v, REPS, sizeof(double), cmp);
    *med = v[REPS / 2];
    printf("%-12s median %.4f ns  IQR [%.4f, %.4f] ns  min %.4f  max %.4f\n", name,
           1e9 * v[REPS / 2], 1e9 * v[REPS / 4], 1e9 * v[3 * REPS / 4], 1e9 * v[0], 1e9 * v[REPS - 1]);
}

int main(void) {
    SetThreadAffinityMask(GetCurrentThread(), 1);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    double la, lm, ta, tm;
    printf("NOPS = %lld per rep, REPS = %d, double precision\n", NOPS, REPS);
    run("lat_add", lat_add, &la);
    run("lat_mul", lat_mul, &lm);
    run("thr_add", thr_add, &ta);
    run("thr_mul", thr_mul, &tm);
    printf("rho_latency = t_mul/t_add = %.4f\n", lm / la);
    printf("rho_throughput = %.4f\n", tm / ta);
    return 0;
}
