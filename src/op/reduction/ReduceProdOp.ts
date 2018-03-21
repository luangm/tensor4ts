import ReductionOp from "./ReductionOp";

export default class ReduceProdOp extends ReductionOp {

  get initialValue() {
    return 1;
  }

  update(accum: number, a: number): number {
    return accum * a;
  }

}