import ReductionOp from "./ReductionOp";
import Tensor from "../../Tensor";

export default class ReduceMeanOp extends ReductionOp {

  get shouldPostProcess() {
    return true;
  }

  constructor(base: Tensor, result: Tensor, reducedDims: boolean[]) {
    super(base, result, reducedDims);
  }

  getResult(accum: number, n: number): number {
    return accum / n;
  }

  update(accum: number, a: number): number {
    return accum + a;
  }

}