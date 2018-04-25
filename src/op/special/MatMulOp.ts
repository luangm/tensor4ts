import Gemm from "../../blas/Gemm";
import Tensor from "../../Tensor";
import Operation from "../Operation";

export default class MatMulOp extends Operation {

  private readonly _left: Tensor;
  private readonly _program: Gemm;
  private readonly _right: Tensor;

  get isSpecial() {
    return true;
  }

  get left() {
    return this._left;
  }

  get right() {
    return this._right;
  }

  constructor(left: Tensor, right: Tensor, result: Tensor, transposeLeft = false, transposeRight = false) {
    super([left, right], [result]);

    this._left = left;
    this._right = right;

    let m = transposeLeft ? left.shape[1] : left.shape[0];
    let n = transposeRight ? right.shape[0] : right.shape[1];
    let k = transposeLeft ? left.shape[0] : left.shape[1];

    this._program = new Gemm(transposeLeft, transposeRight, m, n, k, 1, left.data as Float32Array, 0, right.data as Float32Array, 0, 0, result.data as Float32Array, 0);
  }

  exec() {
    this._program.exec();
  }
}