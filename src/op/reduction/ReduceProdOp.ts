import ReductionOp from "./ReductionOp";
import Tensor from "../../Tensor";

export default class ReduceProdOp extends ReductionOp {

  get initialValue() {
    return 1;
  }

  constructor(base: Tensor, result: Tensor, reducedDims: boolean[]) {
    super(base, result, reducedDims);
  }

  update(accum: number, a: number): number {
    return accum * a;
  }

}