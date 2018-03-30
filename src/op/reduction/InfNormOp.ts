import ReductionOp from "./ReductionOp";

export default class InfNormOp extends ReductionOp {

  body(a: number, b?: number): number {
    return Math.abs(a);
  }

  update(accum: number, a: number): number {
    return Math.max(accum, a);
  }

}