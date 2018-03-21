import TransformOp from "./TransformOp";

export default class SqrtOp extends TransformOp {

  body(a: number, b?: number): number {
    return Math.sqrt(a);
  }

}