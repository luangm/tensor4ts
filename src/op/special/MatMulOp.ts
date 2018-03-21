import Operation from "../Operation";
import Tensor from "../../Tensor";
import Gemm from "../../blas/Gemm";

export default class MatMulOp extends Operation {

  private _program: Gemm;

  constructor(input: Tensor, other: Tensor, result: Tensor, transposeA = false, transposeB = false) {
    super(input, other, result);

    let m = transposeA ? input.shape[1] : input.shape[0];
    let n = transposeB ? other.shape[0] : other.shape[1];
    let k = transposeA ? input.shape[0] : input.shape[1];

    this._program = new Gemm(transposeA, transposeB, m, n, k, 1, input.data, null, other.data, null, 0, result.data, null);
  }

  get isSpecial() {
    return true;
  }

  body(a: number, b?: number): number {
    return 0;
  }

  exec() {
    this._program.exec();
  }
}