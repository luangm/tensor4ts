export default class Gemm {

  private _A: Float32Array;
  private _B: Float32Array;
  private _C: Float32Array;
  private _alpha: number;
  private _beta: number;
  private _k: number;
  private _lda: number;
  private _ldb: number;
  private _ldc: number;
  private _m: number;
  private _n: number;
  private _transA: boolean;
  private _transB: boolean;

  constructor(transA: boolean, transB: boolean, m: number, n: number, k: number,
              alpha: number, A: Float32Array, lda: number, B: Float32Array, ldb: number,
              beta: number, C: Float32Array, ldc: number) {
    this._transA = transA;
    this._transB = transB;
    this._m = m;
    this._n = n;
    this._k = k;
    this._alpha = alpha;
    this._A = A;
    this._B = B;
    this._C = C;
    this._beta = beta;
    this._lda = lda;
    this._ldb = ldb;
    this._ldc = ldc;
  }

  exec() {
    if (!this._transB) {
      if (!this._transA) {
        Gemm.gemmAB(this._m, this._n, this._k, this._alpha, this._A, this._B, this._beta, this._C);
      } else {
        Gemm.gemmAtB(this._m, this._n, this._k, this._alpha, this._A, this._B, this._beta, this._C);
      }
    } else {
      if (!this._transA) {
        Gemm.gemmABt(this._m, this._n, this._k, this._alpha, this._A, this._B, this._beta, this._C);
      } else {
        Gemm.gemmAtBt(this._m, this._n, this._k, this._alpha, this._A, this._B, this._beta, this._C);
      }
    }
  }

  // C = alpha * A * B + beta * C
  private static gemmAB(m: number, n: number, k: number, alpha: number, A: Float32Array, B: Float32Array, beta: number, C: Float32Array) {
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        let temp = 0;
        for (let l = 0; l < k; l++) {
          temp += A[i * k + l] * B[l * n + j];
        }
        let cIndex = i * n + j;
        C[cIndex] = alpha * temp + (beta !== 0 ? beta * C[cIndex] : 0);
      }
    }
  }

  // C = alpha * A * B^T + beta * C
  private static gemmABt(m: number, n: number, k: number, alpha: number, A: Float32Array, B: Float32Array, beta: number, C: Float32Array) {
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        let temp = 0;
        for (let l = 0; l < k; l++) {
          temp += A[i * k + l] * B[j * k + l];
        }
        let cIndex = i * k + j;
        C[cIndex] = alpha * temp + (beta !== 0 ? beta * C[cIndex] : 0);
      }
    }
  }

  // C = alpha * A^T * B + beta * C
  private static gemmAtB(m: number, n: number, k: number, alpha: number, A: Float32Array, B: Float32Array, beta: number, C: Float32Array) {
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        let temp = 0;
        for (let l = 0; l < k; l++) {
          temp += A[l * m + i] * B[l * n + j];
        }
        let cIndex = i * k + j;
        C[cIndex] = alpha * temp + (beta !== 0 ? beta * C[cIndex] : 0);
      }
    }
  }

  // C = alpha * A^T * B^T + beta * C
  private static gemmAtBt(m: number, n: number, k: number, alpha: number, A: Float32Array, B: Float32Array, beta: number, C: Float32Array) {
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        let temp = 0;
        for (let l = 0; l < k; l++) {
          temp += A[l * m + i] * B[j * k + l];
        }
        let index = i * m + j;
        C[index] = alpha * temp + (beta !== 0 ? beta * C[index] : 0);
      }
    }
  }
}