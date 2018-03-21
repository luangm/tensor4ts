import Operation from "../Operation";
import Tensor from "../../Tensor";

export default abstract class ReductionOp extends Operation {

  private _reducedDims: boolean[];

  constructor(input: Tensor, other: Tensor, result: Tensor, reducedDims: boolean[]) {
    super(input, other, result);
    this._reducedDims = reducedDims;
  }

  get initialValue() {
    return 0;
  }

  get reducedDims() {
    return this._reducedDims;
  }

  get shouldPostProcess() {
    return false;
  }

  body(a: number, b?: number): number {
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
