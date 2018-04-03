import Tensor from "../../Tensor";
import ReductionOp from "./ReductionOp";

export default class PNormOp extends ReductionOp {

  private readonly _p: number;

  get p() {
    return this._p;
  }

  get shouldPostProcess() {
    return true;
  }

  constructor(input: Tensor, result: Tensor, p: number = 2, reducedDims: boolean[]) {
    super(input, result, reducedDims);
    this._p = p;
  }

  body(a: number): number {
    return Math.pow(Math.abs(a), this.p);
  }

  getResult(accum: number, n: number): number {
    return Math.pow(accum, 1 / this.p);
  }

  update(accum: number, a: number): number {
    return accum + a;
  }

}