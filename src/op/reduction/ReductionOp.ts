import Operation from "../Operation";
import Tensor from "../../Tensor";

export default abstract class ReductionOp extends Operation {

  private readonly _base: Tensor;
  private readonly _reducedDims: boolean[];
  private readonly _result: Tensor;

  get base() {
    return this._base;
  }

  get initialValue() {
    return 0;
  }

  get reducedDims() {
    return this._reducedDims;
  }

  get result() {
    return this._result;
  }

  get shouldPostProcess() {
    return false;
  }

  protected constructor(base: Tensor, result: Tensor, reducedDims: boolean[]) {
    super([base], [result]);
    this._base = base;
    this._result = result;
    this._reducedDims = reducedDims;
  }

  body(a: number): number {
    return a;
  }

  exec(dim?: number): void {
    // nothing
  }

  getResult(accum: number, n: number): number {
    return accum;
  }

  abstract update(accum: number, a: number): number;
}
