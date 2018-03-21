import TransformOp from "./TransformOp";

export default class SignOp extends TransformOp {

  body(a: number, b?: number): number {
    return Math.sign(a);
  }

}