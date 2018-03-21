import ReductionOp from "./ReductionOp";

export default class ReduceMaxOp extends ReductionOp {

  get initialValue() {
    return -Number.MAX_SAFE_INTEGER;
  }

  update(accum: number, a: number): number {
    return Math.max(accum, a);
  }

}