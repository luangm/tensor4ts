import ReductionOp from "./ReductionOp";

export default class L1NormOp extends ReductionOp {

  body(a: number, b?: number): number {
    return Math.abs(a);
  }

  update(accum: number, a: number): number {
    return accum + a;
  }

}