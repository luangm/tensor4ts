import Operation from "../Operation";
import Tensor from "../../Tensor";

export default abstract class IndexOp extends Operation {

  private readonly _base: Tensor;
  private readonly _dim: number;
  private readonly _result: Tensor;

  get base() {
    return this._base;
  }

  get dim() {
    return this._dim;
  }

  get result() {
    return this._result;
  }

  protected constructor(base: Tensor, result: Tensor, dim: number) {
    super([base], [result]);
    this._base = base;
    this._result = result;
    this._dim = dim;
  }

  body(a: number): number {
    return a;
  }

  exec(dim?: number): void {
    // nothing
  }

  abstract update(accum: number, a: number, accumIdx: number, idx: number): [number, number];
}
