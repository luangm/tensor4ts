import TransformOp from "./TransformOp";

export default class AbsOp extends TransformOp {

  body(a: number, b?: number): number {
    return Math.abs(a);
  }

}