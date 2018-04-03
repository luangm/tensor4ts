import ReductionOp from "./ReductionOp";
import Tensor from "../../Tensor";

export default class L2NormOp extends ReductionOp {

  get shouldPostProcess() {
    return true;
  }

  constructor(base: Tensor, result: Tensor, reducedDims: boolean[]) {
    super(base, result, reducedDims);
  }

  body(a: number): number {
    return a * a;
  }

  getResult(accum: number, n: number): number {
    return Math.sqrt(accum);
  }

  update(accum: number, a: number): number {
    return accum + a;
  }

}