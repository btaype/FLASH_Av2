from common_flash import add_common_args, load_flash_model, print_loaded


def main() -> None:
    parser = add_common_args("Load Llama 3 with FlashAttention 2.")
    parser.add_argument("--no-flash-attn", action="store_true", help="Use the model default attention.")
    parser.add_argument(
        "--attention-backend",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Override attention backend. Use sdpa when flash-attn is not installed.",
    )
    args = parser.parse_args()
    use_flash_attention = not args.no_flash_attn
    _, model = load_flash_model(
        args.model_id,
        use_flash_attention=use_flash_attention,
        attention_backend=args.attention_backend,
    )
    print_loaded(args.model_id, model, use_flash_attention=use_flash_attention, attention_backend=args.attention_backend)


if __name__ == "__main__":
    main()
